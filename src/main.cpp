#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <thread>
#include <future>
#include "app_utils.hpp"
#include "ndarray.hpp"
#include "physics.hpp"
#include "patches.hpp"
#include "ufunc.hpp"
#include "visit_struct.hpp"

using namespace patches2d;




// ============================================================================
#define MIN3ABS(a, b, c) std::min(std::min(std::fabs(a), std::fabs(b)), std::fabs(c))
#define SGN(x) std::copysign(1, x)

static double minmod(double ul, double u0, double ur, double theta)
{
    const double a = theta * (u0 - ul);
    const double b =   0.5 * (ur - ul);
    const double c = theta * (ur - u0);
    return 0.25 * std::fabs(SGN(a) + SGN(b)) * (SGN(a) + SGN(c)) * MIN3ABS(a, b, c);
}




// ============================================================================
struct gradient_plm
{
    gradient_plm(double theta) : theta(theta) {}

    double inline operator()(double a, double b, double c) const
    {
        return minmod(a, b, c, theta);
    }
    double theta;
};

struct atmosphere
{
    inline std::array<double, 5> operator()(std::array<double, 2> X) const
    {
        auto r = X[0];
        // return std::array<double, 5>{std::pow(r, -2), 0.0, 0.0, 0.0, std::pow(r, -2)};
        return std::array<double, 5>{r < 2.0 ? 1.0 : 0.1, 0.0, 0.0, 0.0, r < 2.0 ? 1.0 : 0.125};
        // return std::array<double, 5>{1.0, 0.0, 0.0, 0.0, 1.0};
    }
};




// ============================================================================
void write_database(const Database& database)
{
    auto parts = std::vector<std::string>{"data", "chkpt.0000.jic"};
    filesystem::remove_recurse(filesystem::join(parts));

    for (const auto& patch : database)
    {
        parts.push_back(to_string(patch.first));
        filesystem::require_dir(filesystem::parent(filesystem::join(parts)));
        nd::tofile(patch.second, filesystem::join(parts));
        parts.pop_back();
    }
    std::cout << "Write checkpoint " << filesystem::join(parts) << std::endl;
}

template <typename ConsToPrim>
void write_database_primitive(const Database& database, ConsToPrim cons_to_prim)
{
    auto parts = std::vector<std::string>{"data", "prim.0000.jic"};
    filesystem::remove_recurse(filesystem::join(parts));

    for (const auto& patch : database.all(Field::conserved))
    {
        parts.push_back(to_string(patch.first, "primitive"));
        filesystem::require_dir(filesystem::parent(filesystem::join(parts)));
        nd::tofile(cons_to_prim(patch.second), filesystem::join(parts));
        parts.pop_back();
    }

    for (const auto& patch : database.all(Field::vert_coords))
    {
        parts.push_back(to_string(patch.first));
        filesystem::require_dir(filesystem::parent(filesystem::join(parts)));
        nd::tofile(patch.second, filesystem::join(parts));
        parts.pop_back();
    }

    std::cout << "Write primitives " << filesystem::join(parts) << std::endl;
}




// ============================================================================
struct run_config
{
    void print(std::ostream& os) const;
    static run_config from_dict(std::map<std::string, std::string> items);
    static run_config from_argv(int argc, const char* argv[]);

    std::string outdir = ".";
    double tfinal = 0.0;
    int rk = 1;
    int nr = 32;
    int num_levels = 1;
    int threaded = 0;
};

VISITABLE_STRUCT(run_config,
    outdir,
    tfinal,
    rk,
    nr,
    num_levels,
    threaded);




// ============================================================================
run_config run_config::from_dict(std::map<std::string, std::string> items)
{
    run_config cfg;

    visit_struct::for_each(cfg, [items] (const char* name, auto& value)
    {
        if (items.find(name) != items.end())
        {
            cmdline::set_from_string(items.at(name), value);
        }
    });

    for (const auto& item : items)
    {
        bool found = false;

        visit_struct::for_each(cfg, [item, &found] (const char* name, auto&)
        {
            if (item.first == name)
            {
                found = true;;
            }
        });

        if (! found)
        {
            throw std::runtime_error("unrecognized option: " + item.first);
        }
    }
    return cfg;
}

run_config run_config::from_argv(int argc, const char* argv[])
{
    return from_dict(cmdline::parse_keyval(argc, argv));
}

void run_config::print(std::ostream& os) const
{
    using std::left;
    using std::setw;
    using std::setfill;
    using std::showpos;
    const int W = 24;

    os << std::string(52, '=') << "\n";
    os << "Config:\n\n";

    std::ios orig(nullptr);
    orig.copyfmt(os);

    visit_struct::for_each(*this, [&os] (std::string name, auto& value)
    {
        os << left << setw(W) << setfill('.') << name + " " << " " << value << "\n";
    });

    os << "\n";
    os.copyfmt(orig);
}




// ============================================================================
struct mesh_geometry
{
    nd::array<double, 3> centroids;
    nd::array<double, 3> volumes;
    nd::array<double, 3> face_areas_i;
    nd::array<double, 3> face_areas_j;
};




// ============================================================================
nd::array<double, 3> mesh_vertices(int ni, int nj, std::array<double, 4> extent)
{
    auto X = nd::array<double, 3>(ni + 1, nj + 1, 2);
    auto x0 = extent[0];
    auto x1 = extent[1];
    auto y0 = extent[2];
    auto y1 = extent[3];

    for (int i = 0; i < ni + 1; ++i)
    {
        for (int j = 0; j < nj + 1; ++j)
        {
            X(i, j, 0) = x0 * std::pow(x1 / x0, double(i) / ni);
            X(i, j, 1) = y0 + (y1 - y0) * j / nj;
        }
    }
    return X;
}

nd::array<double, 3> mesh_cell_centroids(const nd::array<double, 3>& verts)
{
    auto centroid_r = ufunc::from([] (double r0, double r1)
    {
        return std::sqrt(r0 * r1);
    });
    auto centroid_q = ufunc::from([] (double q0, double q1)
    {
        return 0.5 * (q0 + q1);
    });

    auto _ = nd::axis::all();
    auto mi = verts.shape(0);
    auto mj = verts.shape(1);
    auto r0 = verts.select(_|0|mi-1, _|0|mj-1, _|0|1);
    auto r1 = verts.select(_|1|mi-0, _|1|mj-0, _|0|1);
    auto q0 = verts.select(_|0|mi-1, _|0|mj-1, _|1|2);
    auto q1 = verts.select(_|1|mi-0, _|1|mj-0, _|1|2);
    auto res = nd::array<double, 3>(mi - 1, mj - 1, 2);

    res.select(_, _, _|0|1) = centroid_r(r0, r1);
    res.select(_, _, _|1|2) = centroid_q(q0, q1);

    return res;
}

nd::array<double, 3> mesh_cell_volumes(const nd::array<double, 3>& verts)
{
    auto _ = nd::axis::all();
    auto p1 = 2 * M_PI;
    auto p0 = 0;
    auto mi = verts.shape(0);
    auto mj = verts.shape(1);
    auto r0 = verts.select(_|0|mi-1, _|0|mj-1, _|0|1);
    auto r1 = verts.select(_|1|mi-0, _|1|mj-0, _|0|1);
    auto q0 = verts.select(_|0|mi-1, _|0|mj-1, _|1|2);
    auto q1 = verts.select(_|1|mi-0, _|1|mj-0, _|1|2);

    auto volume = ufunc::nfrom([p0, p1] (std::array<double, 4> extent)
    {
        auto r0 = extent[0];
        auto r1 = extent[1];
        auto q0 = extent[2];
        auto q1 = extent[3];
        return -1. / 3 * (r1 * r1 * r1 - r0 * r0 * r0) * (std::cos(q1) - std::cos(q0)) * (p1 - p0);
    });

    auto args = std::array<nd::array<double, 3>, 4>{r0, r1, q0, q1};
    return volume(args);
}

nd::array<double, 3> mesh_face_areas_i(const nd::array<double, 3>& verts)
{
    auto _ = nd::axis::all();
    auto p1 = 2 * M_PI;
    auto p0 = 0;
    // auto mi = verts.shape(0);
    auto mj = verts.shape(1);
    auto r0 = verts.select(_, _|0|mj-1, _|0|1);
    auto r1 = verts.select(_, _|1|mj-0, _|0|1);
    auto q0 = verts.select(_, _|0|mj-1, _|1|2);
    auto q1 = verts.select(_, _|1|mj-0, _|1|2);

    auto area = ufunc::nfrom([p0, p1] (std::array<double, 4> extent)
    {
        auto r0 = extent[0];
        // auto r1 = extent[1];
        auto q0 = extent[2];
        auto q1 = extent[3];
        return -r0 * r0 * (p1 - p0) * (std::cos(q1) - std::cos(q0));
    });

    auto args = std::array<nd::array<double, 3>, 4>{r0, r1, q0, q1};
    return area(args);
}

nd::array<double, 3> mesh_face_areas_j(const nd::array<double, 3>& verts)
{
    auto _ = nd::axis::all();
    auto p1 = 2 * M_PI;
    auto p0 = 0;
    auto mi = verts.shape(0);
    // auto mj = verts.shape(1);
    auto r0 = verts.select(_|0|mi-1, _, _|0|1);
    auto r1 = verts.select(_|1|mi-0, _, _|0|1);
    auto q0 = verts.select(_|0|mi-1, _, _|1|2);
    auto q1 = verts.select(_|1|mi-0, _, _|1|2);

    auto area = ufunc::nfrom([p0, p1] (std::array<double, 4> extent)
    {
        auto r0 = extent[0];
        auto r1 = extent[1];
        auto q0 = extent[2];
        // auto q1 = extent[3];
        return 0.5 * (r1 + r0) * (r1 - r0) * (p1 - p0) * std::sin(q0);
    });

    auto args = std::array<nd::array<double, 3>, 4>{r0, r1, q0, q1};
    return area(args);
}




// ============================================================================
nd::array<double, 3> pad_with_zeros_j(const nd::array<double, 3>& A)
{
    auto _ = nd::axis::all();
    auto ni = A.shape(0);
    auto nj = A.shape(1);
    auto nk = A.shape(2);

    auto res = nd::ndarray<double, 3>(ni, nj + 2, nk);
    res.select(_, _|1|nj+1, _) = A;
    return res;
}




// ============================================================================
auto advance_2d(nd::array<double, 3> U0, const mesh_geometry& G, double dt)
{
    auto _ = nd::axis::all();

    auto update_formula = [dt] (std::array<double, 5> s, std::array<double, 5> df, std::array<double, 1> dv)
    {
        return std::array<double, 5>{
            dt * (s[0] - df[0] / dv[0]),
            dt * (s[1] - df[1] / dv[0]),
            dt * (s[2] - df[2] / dv[0]),
            dt * (s[3] - df[3] / dv[0]),
            dt * (s[4] - df[4] / dv[0]),
        };
    };

    auto flux_times_area_formula = [] (std::array<double, 5> f, std::array<double, 1> da)
    {
        return std::array<double, 5>{
            f[0] * da[0],
            f[1] * da[0],
            f[2] * da[0],
            f[3] * da[0],
            f[4] * da[0],
        };
    };

    auto gradient_est = ufunc::from(gradient_plm(2.0));
    auto advance_cons = ufunc::vfrom(update_formula);
    auto evaluate_src = ufunc::vfrom(newtonian_hydro::sph_geom_src_terms());
    auto cons_to_prim = ufunc::vfrom(newtonian_hydro::cons_to_prim());
    auto godunov_flux_i = ufunc::vfrom(newtonian_hydro::riemann_hlle({1, 0, 0}));
    auto godunov_flux_j = ufunc::vfrom(newtonian_hydro::riemann_hlle({0, 1, 0}));
    auto extrap_l = ufunc::from([] (double a, double b) { return a - b * 0.5; });
    auto extrap_r = ufunc::from([] (double a, double b) { return a + b * 0.5; });
    auto flux_times_area = ufunc::vfrom(flux_times_area_formula);

    auto mi = U0.shape(0);
    auto mj = U0.shape(1);
    auto P0 = cons_to_prim(U0);

    auto Fhi = [&] ()
    {
        auto Pa = P0.select(_|0|mi-2, _, _);
        auto Pb = P0.select(_|1|mi-1, _, _);
        auto Pc = P0.select(_|2|mi-0, _, _);
        auto Gb = gradient_est(Pa, Pb, Pc);
        auto Pl = extrap_l(Pb, Gb);
        auto Pr = extrap_r(Pb, Gb);
        auto Fh = godunov_flux_i(Pr.take<0>(_|0|mi-3), Pl.take<0>(_|1|mi-2));
        auto Fa = flux_times_area(Fh, G.face_areas_i);
        return Fa;
    }();

    auto Fhj = [&] ()
    {
        auto Pa = P0.select(_|2|mi-2, _|0|mj-2, _);
        auto Pb = P0.select(_|2|mi-2, _|1|mj-1, _);
        auto Pc = P0.select(_|2|mi-2, _|2|mj-0, _);
        auto Gb = pad_with_zeros_j(gradient_est(Pa, Pb, Pc));
        auto Pl = extrap_l(P0.take<0>(_|2|mi-2), Gb);
        auto Pr = extrap_r(P0.take<0>(_|2|mi-2), Gb);
        auto Fh = pad_with_zeros_j(godunov_flux_j(Pr.take<1>(_|0|mj-1), Pl.take<1>(_|1|mj)));
        auto Fa = flux_times_area(Fh, G.face_areas_j);
        return Fa;
    }();

    auto dFi = Fhi.take<0>(_|1|mi-3) - Fhi.take<0>(_|0|mi-4);
    auto dFj = Fhj.take<1>(_|1|mj+1) - Fhj.take<1>(_|0|mj+0);
    auto dF = dFi + dFj;

    auto S0 = evaluate_src(P0.take<0>(_|2|mi-2), G.centroids);
    auto dU = advance_cons(S0, dF, G.volumes);

    return U0.take<0>(_|2|mi-2) + dU;
}




// ============================================================================
void update_2d_nothread(Database& database, double dt, double rk_factor)
{
    auto results = std::map<Database::Index, Database::Array>();

    for (const auto& patch : database.all(Field::conserved))
    {
        auto U = database.fetch(patch.first, 2, 2, 0, 0);
        auto G = mesh_geometry();

        G.centroids   .become(database.at(patch.first, Field::cell_coords));
        G.volumes     .become(database.at(patch.first, Field::cell_volume));
        G.face_areas_i.become(database.at(patch.first, Field::face_area_i));
        G.face_areas_j.become(database.at(patch.first, Field::face_area_j));

        results[patch.first].become(advance_2d(U, G, dt));
    }
    for (const auto& res : results)
    {
        database.commit(res.first, res.second, rk_factor);
    }
}

void update(Database& database, double dt, int rk, int /*threaded*/)
{
    auto up = update_2d_nothread;

    switch (rk)
    {
        case 1:
            up(database, dt, 0.0);
            break;
        case 2:
            up(database, dt, 0.0);
            up(database, dt, 0.5);
            break;
        default:
            throw std::invalid_argument("rk must be 1 or 2");
    }
}




// ============================================================================
nd::array<double, 3> boundary_value(
    Database::Index,
    PatchBoundary edge,
    int depth,
    const nd::array<double, 3>& patch)
{
    if (depth != 2)
    {
        throw std::logic_error("boundary_value: should not be here");
    }

    auto _ = nd::axis::all();

    switch (edge)
    {
        case PatchBoundary::il:
        {
            auto res = nd::array<double, 3>(2, patch.shape(1), 5);

            for (int n = 0; n < 5; ++n)
            {
                res.select(0, _, n) = patch.select(0, _, n);
                res.select(1, _, n) = patch.select(0, _, n);
            }
            return res;
        }
        case PatchBoundary::ir:
        {
            auto res = nd::array<double, 3>(2, patch.shape(1), 5);

            res.select(0, _, _) = patch.select(patch.shape(0) - 1, _, _);
            res.select(1, _, _) = patch.select(patch.shape(0) - 1, _, _);

            return res;
        }
        case PatchBoundary::jl: throw std::logic_error("boundary_value: should not be here");
        case PatchBoundary::jr: throw std::logic_error("boundary_value: should not be here");
    }
}




// ============================================================================
Database build_database(int ni, int nj)
{
    auto header = Database::Header
    {
        {Field::conserved,   {5, MeshLocation::cell}},
        {Field::vert_coords, {2, MeshLocation::vert}},
        {Field::cell_coords, {2, MeshLocation::cell}},
        {Field::cell_volume, {1, MeshLocation::cell}},
        {Field::face_area_i, {1, MeshLocation::face_i}},
        {Field::face_area_j, {1, MeshLocation::face_j}},
    };

    auto database = Database(ni, nj, header);
    auto initial_data = ufunc::vfrom(atmosphere());
    auto prim_to_cons = ufunc::vfrom(newtonian_hydro::prim_to_cons());

    auto x_verts = mesh_vertices(ni, nj, {1, 10, 0, M_PI});
    auto x_cells = mesh_cell_centroids(x_verts);
    auto v_cells = mesh_cell_volumes(x_verts);
    auto a_faces_i = mesh_face_areas_i(x_verts);
    auto a_faces_j = mesh_face_areas_j(x_verts);
    auto U = prim_to_cons(initial_data(x_cells));

    database.insert(std::make_tuple(0, 0, 0, Field::vert_coords), x_verts);
    database.insert(std::make_tuple(0, 0, 0, Field::cell_coords), x_cells);
    database.insert(std::make_tuple(0, 0, 0, Field::cell_volume), v_cells);
    database.insert(std::make_tuple(0, 0, 0, Field::face_area_i), a_faces_i);
    database.insert(std::make_tuple(0, 0, 0, Field::face_area_j), a_faces_j);
    database.insert(std::make_tuple(0, 0, 0, Field::conserved), U);

    database.set_boundary_value(boundary_value);
    return database;
}




// ============================================================================
int main_2d(int argc, const char* argv[])
{
    auto cfg  = run_config::from_argv(argc, argv);
    auto wall = 0.0;
    auto ni   = cfg.nr;
    auto nj   = cfg.nr;
    auto iter = 0;
    auto t    = 0.0;
    auto dt   = 0.001; // TODO
    auto database = build_database(ni, nj);


    // ========================================================================
    std::cout << "\n";
    cfg.print(std::cout);
    database.print(std::cout);
    std::cout << std::string(52, '=') << "\n";
    std::cout << "Main loop:\n\n";


    // ========================================================================
    // Main loop
    // ========================================================================
    while (t < cfg.tfinal)
    {
        auto timer = Timer();
        update(database, dt, cfg.rk, cfg.threaded);

        t    += dt;
        iter += 1;
        wall += timer.seconds();
        auto kzps = database.num_cells(Field::conserved) / 1e3 / timer.seconds();

        std::printf("[%04d] t=%3.2lf kzps=%3.2lf\n", iter, t, kzps);
    }

    std::printf("average kzps=%f\n", database.num_cells(Field::conserved) / 1e3 / wall * iter);
    // write_database(database);
    write_database_primitive(database, ufunc::vfrom(newtonian_hydro::cons_to_prim()));

    return 0;
}




// ============================================================================
int main(int argc, const char* argv[])
{
    std::set_terminate(debug::terminate_with_backtrace);

    // return main_2d(argc, argv);

    try {
        return main_2d(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << "\nERROR: ";
        std::cerr << e.what() << "\n\n";
        return 1;
    }
}
