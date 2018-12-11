#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
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

struct gradient_plm
{
    gradient_plm(double theta) : theta(theta) {}

    double inline operator()(double a, double b, double c) const
    {
        return minmod(a, b, c, theta);
    }
    double theta;
};




// ============================================================================
void write_chkpt(const Database& database, std::string filename)
{
    std::cout << "write checkpoint " << filename << std::endl;
    auto parts = std::vector<std::string>{filename};

    filesystem::remove_recurse(filesystem::join(parts));

    for (const auto& patch : database)
    {
        parts.push_back(to_string(patch.first));
        filesystem::require_dir(filesystem::parent(filesystem::join(parts)));
        nd::tofile(patch.second, filesystem::join(parts));
        parts.pop_back();
    }
}

void write_primitive(const Database& database, std::string filename)
{
    std::cout << "write primitive " << filename << std::endl;
    auto parts = std::vector<std::string>{filename};
    auto cons_to_prim = ufunc::vfrom(newtonian_hydro::cons_to_prim());

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
    std::cout << "write primitives " << filesystem::join(parts) << std::endl;
}

void write_vtk(const Database& database, std::string filename)
{
    std::cout << "write VTK " << filename << std::endl;
    filesystem::require_dir(filesystem::parent(filename));

    auto cons_to_prim = ufunc::vfrom(newtonian_hydro::cons_to_prim());
    auto stream = std::fstream(filename, std::ios::out);
    auto vert = database.at(std::make_tuple(0, 0, 0, Field::vert_coords));


    // ------------------------------------------------------------------------
    // Write header
    // ------------------------------------------------------------------------
    stream << "# vtk DataFile Version 3.0\n";
    stream << "My Data" << "\n";
    stream << "ASCII\n";
    stream << "DATASET STRUCTURED_GRID\n";
    stream << "DIMENSIONS " << vert.shape(0) << " " << vert.shape(1) << " " << 1 << "\n";


    // ------------------------------------------------------------------------
    // Write vertex points
    // ------------------------------------------------------------------------
    stream << "POINTS " << vert.shape(0) * vert.shape(1) << " FLOAT\n";

    for (int j = 0; j < vert.shape(1); ++j)
    {
        for (int i = 0; i < vert.shape(0); ++i)
        {
            const double r = vert(i, j, 0);
            const double q = vert(i, j, 1);
            const double x = r * std::sin(q);
            const double z = r * std::cos(q);
            stream << x << " " << 0.0 << " " << z << "\n";
        }
    }


    // ------------------------------------------------------------------------
    // Write primitive data
    // ------------------------------------------------------------------------
    auto cons = database.at(std::make_tuple(0, 0, 0, Field::conserved));
    auto prim = cons_to_prim(cons);
    stream << "CELL_DATA " << prim.shape(0) * prim.shape(1) << "\n";
    stream << "SCALARS " << "density " << "double " << 1 << "\n";
    stream << "LOOKUP_TABLE default\n";

    for (int j = 0; j < prim.shape(1); ++j)
    {
        for (int i = 0; i < prim.shape(0); ++i)
        {
            stream << prim(i, j, 0) << "\n";
        }
    }

    stream << "SCALARS " << "radial_velocity " << "double " << 1 << "\n";
    stream << "LOOKUP_TABLE default\n";

    for (int j = 0; j < prim.shape(1); ++j)
    {
        for (int i = 0; i < prim.shape(0); ++i)
        {
            stream << prim(i, j, 1) << "\n";
        }
    }
}




// ============================================================================
struct run_config
{
    void print(std::ostream& os) const;
    run_config validate() const;
    std::string make_filename_checkpoint(int count) const;
    std::string make_filename_primitive(int count) const;
    std::string make_filename_vtk(int count) const;
    static run_config from_dict(std::map<std::string, std::string> items);
    static run_config from_argv(int argc, const char* argv[]);

    /** Run control */
    std::string outdir = "data";
    double tfinal      = 0.1;
    int rk             = 1;
    int nr             = 32;
    int num_levels     = 1;
    int threaded       = 0;
    int test_mode      = 0;

    /** Physics setup */
    double jet_opening_angle = 0.5;
    double jet_power         = 1.0;
    double jet_velocity      = 1.0;
    double jet_density       = 1.0;
    double density_index     = 2.0;
    double ambient_pressure  = 0.01;
    double outer_radius      = 10.0;
};

VISITABLE_STRUCT(run_config,
    outdir,
    tfinal,
    rk,
    nr,
    num_levels,
    threaded,
    test_mode,
    jet_opening_angle,
    jet_power,
    jet_velocity,
    jet_density,
    density_index,
    ambient_pressure,
    outer_radius);




// ============================================================================
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
        os << "\t" << left << setw(W) << setfill('.') << name + " " << " " << value << "\n";
    });

    os << "\n";
    os.copyfmt(orig);
}

run_config run_config::validate() const
{
    if (nr < 4)             throw std::runtime_error("nr must be >= 4");
    if (rk != 1 && rk != 2) throw std::runtime_error("rk must be 1 or 2");
    if (outer_radius < 2.0) throw std::runtime_error("outer_radius must be > 2");
    return *this;
}

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

std::string run_config::make_filename_checkpoint(int count) const
{
    auto ss = std::stringstream();
    ss << "chkpt." << std::setfill('0') << std::setw(4) << count;
    return filesystem::join({outdir, ss.str()});
}

std::string run_config::make_filename_primitive(int count) const
{
    auto ss = std::stringstream();
    ss << "primitive." << std::setfill('0') << std::setw(4) << count;
    return filesystem::join({outdir, ss.str()});
}

std::string run_config::make_filename_vtk(int count) const
{
    auto ss = std::stringstream();
    ss << std::setfill('0') << std::setw(4) << count << ".vtk";
    return filesystem::join({outdir, ss.str()});
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

void update(Database& database, double dt, int rk, int threaded)
{
    if (threaded)
    {
        throw std::invalid_argument("threaded update not written yet");
    }
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
struct atmosphere
{
    atmosphere(double density_index, double ambient_pressure)
    : density_index(density_index)
    , ambient_pressure(ambient_pressure)
    {
    }

    inline std::array<double, 5> operator()(std::array<double, 2> X) const
    {
        const double r = X[0];
        const double a = density_index;
        const double p = ambient_pressure;
        return std::array<double, 5>{std::pow(r, -a), 0.0, 0.0, 0.0, p};
    }
    double density_index;
    double ambient_pressure;
};




// ============================================================================
struct explosion
{
    inline std::array<double, 5> operator()(std::array<double, 2> X) const
    {
        double d = X[0] < 2 ? 1.0 : 0.1;
        double p = X[0] < 2 ? 1.0 : 0.125;;
        return std::array<double, 5>{d, 0.0, 0.0, 0.0, p};
    }
};




// ============================================================================
struct jet_boundary_value
{
    jet_boundary_value(run_config cfg) : cfg(cfg) {}

    nd::array<double, 3> operator()(
        Database::Index,
        PatchBoundary edge,
        int /*depth*/,
        const nd::array<double, 3>& patch) const
    {
        switch (edge)
        {
            case PatchBoundary::il: return inflow_inner(patch);
            case PatchBoundary::ir: return zero_gradient_outer(patch);
            case PatchBoundary::jl: return nd::array<double, 3>();
            case PatchBoundary::jr: return nd::array<double, 3>();
        }
    }

    nd::array<double, 3> zero_gradient_outer(const nd::array<double, 3>& patch) const
    {
        auto _ = nd::axis::all();
        auto U = nd::array<double, 3>(2, patch.shape(1), 5);
        U.select(0, _, _) = patch.select(patch.shape(0) - 1, _, _);
        U.select(1, _, _) = patch.select(patch.shape(0) - 1, _, _);
        return U;
    }

    nd::array<double, 3> inflow_inner(const nd::array<double, 3>& patch) const
    {
        auto prim_to_cons = ufunc::vfrom(newtonian_hydro::prim_to_cons());
        auto P = nd::array<double, 3>(2, patch.shape(1), 5);

        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < patch.shape(1); ++j)
            {
                auto q = M_PI * (j + 0.5) / patch.shape(1);
                auto inflowP = jet_inlet(q);

                for (int k = 0; k < 5; ++k)
                {
                    P(i, j, k) = inflowP[k];
                }
            }
        }
        return prim_to_cons(P);
    }

    newtonian_hydro::Vars jet_inlet(double q) const
    {
        auto q0 = 0.0;
        auto q1 = M_PI;
        auto u0 = cfg.jet_velocity;
        auto dg = cfg.jet_density;
        auto dq = cfg.jet_opening_angle;
        auto f0 = u0 * std::exp(-std::pow((q - q0) / dq, 2));
        auto f1 = u0 * std::exp(-std::pow((q - q1) / dq, 2));
        auto inflowP = newtonian_hydro::Vars{dg, f0 + f1, 0.0, 0.0, cfg.ambient_pressure};
        return inflowP;
    }
    run_config cfg;
};




// ============================================================================
struct simple_boundary_value
{
    nd::array<double, 3> operator()(
        Database::Index,
        PatchBoundary edge,
        int /*depth*/,
        const nd::array<double, 3>& patch) const
    {
        switch (edge)
        {
            case PatchBoundary::il: return inflow_inner(patch);
            case PatchBoundary::ir: return zero_gradient_outer(patch);
            case PatchBoundary::jl: return nd::array<double, 3>();
            case PatchBoundary::jr: return nd::array<double, 3>();
        }
    }

    nd::array<double, 3> zero_gradient_outer(const nd::array<double, 3>& patch) const
    {
        auto _ = nd::axis::all();
        auto U = nd::array<double, 3>(2, patch.shape(1), 5);
        U.select(0, _, _) = patch.select(patch.shape(0) - 1, _, _);
        U.select(1, _, _) = patch.select(patch.shape(0) - 1, _, _);
        return U;
    }

    nd::array<double, 3> inflow_inner(const nd::array<double, 3>& patch) const
    {
        auto prim_to_cons = ufunc::vfrom(newtonian_hydro::prim_to_cons());
        auto _ = nd::axis::all();
        auto P = nd::array<double, 3>(2, patch.shape(1), 5);
        P.select(_, _, 0) = 1.0;
        P.select(_, _, 1) = 1.0;
        return prim_to_cons(P);
    }
};




// ============================================================================
Database create_database(run_config cfg)
{
    auto ni = cfg.nr * std::log10(cfg.outer_radius);
    auto nj = cfg.nr;

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
    auto prim_to_cons = ufunc::vfrom(newtonian_hydro::prim_to_cons());
    auto x_verts = mesh_vertices(ni, nj, {1, cfg.outer_radius, 0, M_PI});
    auto x_cells = mesh_cell_centroids(x_verts);
    auto v_cells = mesh_cell_volumes(x_verts);
    auto a_faces_i = mesh_face_areas_i(x_verts);
    auto a_faces_j = mesh_face_areas_j(x_verts);

    database.insert(std::make_tuple(0, 0, 0, Field::vert_coords), x_verts);
    database.insert(std::make_tuple(0, 0, 0, Field::cell_coords), x_cells);
    database.insert(std::make_tuple(0, 0, 0, Field::cell_volume), v_cells);
    database.insert(std::make_tuple(0, 0, 0, Field::face_area_i), a_faces_i);
    database.insert(std::make_tuple(0, 0, 0, Field::face_area_j), a_faces_j);

    if (cfg.test_mode)
    {
        auto initial_data = ufunc::vfrom(explosion());
        database.insert(std::make_tuple(0, 0, 0, Field::conserved), prim_to_cons(initial_data(x_cells)));
        database.set_boundary_value(simple_boundary_value());
    }
    else
    {
        auto initial_data = ufunc::vfrom(atmosphere(cfg.density_index, cfg.ambient_pressure));
        database.insert(std::make_tuple(0, 0, 0, Field::conserved), prim_to_cons(initial_data(x_cells)));
        database.set_boundary_value(jet_boundary_value(cfg));
    }
    return database;
}




// ============================================================================
Scheduler create_scheduler(run_config cfg, const Database& database)
{
    auto scheduler = Scheduler();

    scheduler.repeat("write vtk", 0.1, [&database, cfg] (double, int count)
    {
        write_vtk(database, cfg.make_filename_vtk(count));
    });
    scheduler.repeat("write primitive", 0.5, [&database, cfg] (double, int count)
    {
        write_chkpt(database, cfg.make_filename_primitive(count));
    });
    scheduler.repeat("write checkpoint", 1.0, [&database, cfg] (double, int count)
    {
        write_chkpt(database, cfg.make_filename_checkpoint(count));
    });
    return scheduler;
}




// ============================================================================
int run(int argc, const char* argv[])
{
    auto cfg  = run_config::from_argv(argc, argv).validate();
    auto wall = 0.0;
    auto iter = 0;
    auto time = 0.0;
    auto dt   = 0.25 * M_PI / cfg.nr;
    auto database  = create_database(cfg);
    auto scheduler = create_scheduler(cfg, database);


    // ========================================================================
    std::cout << "\n";
    cfg.print(std::cout);
    database.print(std::cout);
    scheduler.print(std::cout);

    std::cout << std::string(52, '=') << "\n";
    std::cout << "Main loop:\n\n";


    // ========================================================================
    // Main loop
    // ========================================================================
    while (time < cfg.tfinal)
    {
        scheduler.dispatch(time);

        auto timer = Timer();
        update(database, dt, cfg.rk, cfg.threaded);

        time += dt;
        iter += 1;
        wall += timer.seconds();

        auto kzps = database.num_cells(Field::conserved) / 1e3 / timer.seconds();
        std::printf("[%04d] t=%3.3lf kzps=%3.2lf\n", iter, time, kzps);
    }

    scheduler.dispatch(time);


    // ========================================================================
    // Final report
    // ========================================================================
    std::cout << "\n";
    std::cout << std::string(52, '=') << "\n";
    std::cout << "Run completed:\n\n";
    std::printf("\taverage kzps=%f\n", database.num_cells(Field::conserved) / 1e3 / wall * iter);
    std::cout << "\n";

    return 0;
}




// ============================================================================
int main(int argc, const char* argv[])
{
    // std::set_terminate(debug::terminate_with_backtrace);
    // return run(argc, argv);

    try {
        return run(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << "\nERROR: ";
        std::cerr << e.what() << "\n\n";
        return 1;
    }
}
