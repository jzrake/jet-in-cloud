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

struct shocktube
{
    inline std::array<double, 5> operator()(std::array<double, 1> x) const
    {
        return x[0] < 0.5
        ? std::array<double, 5>{1.0, 0.0, 0.0, 0.0, 1.000}
        : std::array<double, 5>{0.1, 0.0, 0.0, 0.0, 0.125};
    }
};

struct shocktube_2d
{
    inline std::array<double, 5> operator()(std::array<double, 2> x) const
    {
        return x[0] + x[1] < 1.0 + 1e-10
        ? std::array<double, 5>{1.0, 0.0, 0.0, 0.0, 1.000}
        : std::array<double, 5>{0.1, 0.0, 0.0, 0.0, 0.125};
    }
};

struct cylindrical_explosion
{
    inline std::array<double, 5> operator()(std::array<double, 2> X) const
    {
        auto x = X[0] - 0.5;
        auto y = X[1] - 0.5;
        return x * x + y * y < 0.05
        ? std::array<double, 5>{1.0, 0.0, 0.0, 0.0, 1.000}
        : std::array<double, 5>{0.1, 0.0, 0.0, 0.0, 0.125};
    }
};

struct gaussian_density
{
    inline std::array<double, 5> operator()(std::array<double, 2> X) const
    {
        auto x = X[0] - 0.25;
        auto y = X[1] - 0.25;
        auto d = 1 + std::exp(-(x * x + y * y) / 0.01);
        return std::array<double, 5>{d, 0.5, 0.5, 0.0, 1.000};
    }
};




// ============================================================================
void write_database(const Database& database)
{
    auto parts = std::vector<std::string>{"data", "chkpt.0000.bt"};

    FileSystem::removeRecursively(FileSystem::joinPath(parts));

    for (const auto& patch : database)
    {
        parts.push_back(to_string(patch.first));
        FileSystem::ensureParentDirectoryExists(FileSystem::joinPath(parts));
        nd::tofile(patch.second, FileSystem::joinPath(parts));
        parts.pop_back();
    }
    std::cout << "Write checkpoint " << FileSystem::joinPath(parts) << std::endl;
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
    int ni = 100;
    int nj = 100;
    int num_levels = 1;
    int threaded = 0;
};

VISITABLE_STRUCT(run_config,
    outdir,
    tfinal,
    rk,
    ni,
    nj,
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
auto advance_2d(nd::array<double, 3> U0, double dt)
{
    auto _ = nd::axis::all();

    auto update_formula = [dt] (std::array<double, 9> arg)
    {
        double u   = arg[0];
        double s   = arg[1];
        double fri = arg[2];
        double fli = arg[3];
        double frj = arg[4];
        double flj = arg[5];
        double dai = arg[6];
        double daj = arg[7];
        double dv  = arg[8];
        return u + dt * (s - (dai * (fri - fli) + daj * (frj - flj)) / dv);
    };

    auto gradient_est = ufunc::from(gradient_plm(2.0));
    auto advance_cons = ufunc::nfrom(update_formula);
    auto evaluate_src = ufunc::vfrom(newtonian_hydro::sph_geom_src_terms());
    auto cons_to_prim = ufunc::vfrom(newtonian_hydro::cons_to_prim());
    auto godunov_flux_i = ufunc::vfrom(newtonian_hydro::riemann_hlle({1, 0, 0}));
    auto godunov_flux_j = ufunc::vfrom(newtonian_hydro::riemann_hlle({0, 1, 0}));
    auto extrap_l = ufunc::from([] (double a, double b) { return a - b * 0.5; });
    auto extrap_r = ufunc::from([] (double a, double b) { return a + b * 0.5; });

    auto mi = U0.shape(0);
    auto mj = U0.shape(1);
    auto P0 = cons_to_prim(U0);

    auto Fhi = [&] ()
    {
        auto Pa = P0.select(_|0|mi-2, _|2|mj-2, _);
        auto Pb = P0.select(_|1|mi-1, _|2|mj-2, _);
        auto Pc = P0.select(_|2|mi-0, _|2|mj-2, _);
        auto Gb = gradient_est(Pa, Pb, Pc);
        auto Pl = extrap_l(Pb, Gb);
        auto Pr = extrap_r(Pb, Gb);
        auto Fh = godunov_flux_i(Pr.take<0>(_|0|mi-3), Pl.take<0>(_|1|mi-2));
        return Fh;
    }();

    auto Fhj = [&] ()
    {
        auto Pa = P0.select(_|2|mi-2, _|0|mj-2, _);
        auto Pb = P0.select(_|2|mi-2, _|1|mj-1, _);
        auto Pc = P0.select(_|2|mi-2, _|2|mj-0, _);
        auto Gb = gradient_est(Pa, Pb, Pc);
        auto Pl = extrap_l(Pb, Gb);
        auto Pr = extrap_r(Pb, Gb);
        auto Fh = godunov_flux_j(Pr.take<1>(_|0|mj-3), Pl.take<1>(_|1|mj-2));
        return Fh;
    }();


    auto X0 = nd::array<double, 3>(P0.shape(0), P0.shape(1), 2); // TODO: put cell-center coords
    auto dV = nd::array<double, 3>(P0.shape(0), P0.shape(1), 2); // TODO: put cell volumes
    auto dAi = nd::array<double, 3>(P0.shape(0), P0.shape(1), 2); // TODO: put face areas
    auto dAj = nd::array<double, 3>(P0.shape(0), P0.shape(1), 2); // TODO: put face areas
    auto S0 = evaluate_src(P0, X0);

    return advance_cons(std::array<nd::array<double, 3>, 9>
    {
        U0.select(_|2|mi-2, _|2|mj-2, _),
        S0.select(_|2|mi-2, _|2|mj-2, _),
        Fhi.take<0>(_|1|mi-3),
        Fhi.take<0>(_|0|mi-4),
        Fhj.take<1>(_|1|mj-3),
        Fhj.take<1>(_|0|mj-4),
        dAi,
        dAj,
        dV,
    });
}




// ============================================================================
void update_2d_nothread(Database& database, double dt, double rk_factor)
{
    auto results = std::map<Database::Index, Database::Array>();

    for (const auto& patch : database.all(Field::conserved))
    {
        auto U = database.fetch(patch.first, 2);
        results[patch.first].become(advance_2d(U, dt));
    }
    for (const auto& res : results)
    {
        database.commit(res.first, res.second, rk_factor);
    }
}

void update_2d_threaded(Database& database, double dt, double rk_factor)
{
    struct ThreadResult
    {
        Database::Index index;
        nd::array<double, 3> U1;
    };

    auto threads = std::vector<std::thread>();
    auto futures = std::vector<std::future<ThreadResult>>();

    for (const auto& patch : database.all(Field::conserved))
    {     
        auto U = database.fetch(patch.first, 2);
        auto promise = std::promise<ThreadResult>();

        futures.push_back(promise.get_future());
        threads.push_back(std::thread([index=patch.first,U,dt] (auto promise)
        {
            promise.set_value({index, advance_2d(U, dt)});
        }, std::move(promise)));
    }

    for (auto& f : futures)
    {
        auto res = f.get();
        database.commit(res.index, res.U1, rk_factor);
    }

    for (auto& t : threads)
    {
        t.join();
    }
}

void update(Database& database, double dt, int rk, int threaded)
{
    auto up = threaded ? update_2d_threaded : update_2d_nothread;

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
            X(i, j, 0) = x0 + (x1 - x0) * i / ni;
            X(i, j, 1) = y0 + (y1 - y0) * j / nj;
        }
    }
    return X;
}

nd::array<double, 3> mesh_cell_coords(nd::array<double, 3> verts)
{
    auto _ = nd::axis::all();
    auto ni = verts.shape(0) - 1;
    auto nj = verts.shape(1) - 1;

    return (
    verts.select(_|0|ni+0, _|0|nj+0, _) +
    verts.select(_|0|ni+0, _|1|nj+1, _) +
    verts.select(_|1|ni+1, _|0|nj+0, _) +
    verts.select(_|1|ni+1, _|1|nj+1, _)) * 0.25;
}

nd::array<double, 3> mesh_cell_volumes(nd::array<double, 3> verts)
{
    auto _ = nd::axis::all();
    auto p1 = 2 * M_PI;
    auto p0 = 0;
    auto ni = verts.shape(0);
    auto nj = verts.shape(1);
    auto r0 = verts.select(_|0|ni-1, _|0|nj-1, _|0|1);
    auto r1 = verts.select(_|1|ni-0, _|1|nj-0, _|0|1);
    auto q0 = verts.select(_|0|ni-1, _|0|nj-1, _|1|2);
    auto q1 = verts.select(_|1|ni-0, _|1|nj-0, _|1|2);

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

nd::array<double, 3> mesh_face_areas_i(nd::array<double, 3> verts)
{
    auto _ = nd::axis::all();
    auto p1 = 2 * M_PI;
    auto p0 = 0;
    auto ni = verts.shape(0);
    auto nj = verts.shape(1);
    auto r0 = verts.select(_|0|ni-1, _|0|nj-1, _|0|1);
    auto r1 = verts.select(_|1|ni-0, _|1|nj-0, _|0|1);
    auto q0 = verts.select(_|0|ni-1, _|0|nj-1, _|1|2);
    auto q1 = verts.select(_|1|ni-0, _|1|nj-0, _|1|2);

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

// nd::array<double, 3> mesh_face_areas_j(nd::array<double, 3> verts)
// {
//     auto _ = nd::axis::all();
//     auto p1 = 2 * M_PI;
//     auto p0 = 0;
//     auto ni = verts.shape(0);
//     auto nj = verts.shape(1);
//     auto r0 = verts.select(_|0|ni-1, _|0|nj-1, _|0|1);
//     auto r1 = verts.select(_|1|ni-0, _|1|nj-0, _|0|1);
//     auto q0 = verts.select(_|0|ni-1, _|0|nj-1, _|1|2);
//     auto q1 = verts.select(_|1|ni-0, _|1|nj-0, _|1|2);

//     auto area = ufunc::nfrom([p0, p1] (std::array<double, 4> extent)
//     {
//         auto r0 = extent[0];
//         auto r1 = extent[1];
//         // auto q0 = extent[2];
//         auto q1 = extent[3];
//         return 1. / 3 * (r1 * r1 * r1 - r0 * r0 * r0) * std::sin(q0) * (p1 - p0);
//     });

//     auto args = std::array<nd::array<double, 3>, 4>{r0, r1, q0, q1};
//     return area(args);
// }


// ============================================================================
Database build_database(int ni, int nj)
{
    auto header = Database::Header
    {
        {Field::conserved,    {5, MeshLocation::cell}},
        {Field::vert_coords,  {2, MeshLocation::vert}},
        {Field::cell_coords,  {2, MeshLocation::cell}},
        {Field::cell_volume,  {1, MeshLocation::cell}},
        {Field::face_areas_i, {1, MeshLocation::face_i}},
        {Field::face_areas_j, {1, MeshLocation::face_j}},
    };

    auto database = Database(ni, nj, header);
    auto initial_data = ufunc::vfrom(cylindrical_explosion());
    auto prim_to_cons = ufunc::vfrom(newtonian_hydro::prim_to_cons());

    auto x_verts = mesh_vertices(ni, nj, {1, 10, 0, 2 * M_PI});
    auto x_cells = mesh_cell_coords(x_verts);
    auto v_cells = mesh_cell_volumes(x_verts);
    auto U = prim_to_cons(initial_data(x_cells));

    database.insert(std::make_tuple(0, 0, 0, Field::vert_coords), x_verts);
    database.insert(std::make_tuple(0, 0, 0, Field::cell_coords), x_cells);
    database.insert(std::make_tuple(0, 0, 0, Field::cell_volume), v_cells);
    database.insert(std::make_tuple(0, 0, 0, Field::conserved), U);

    return database;
}




// ============================================================================
int main_2d(int argc, const char* argv[])
{
    auto cfg  = run_config::from_argv(argc, argv);
    auto wall = 0.0;
    auto ni   = cfg.ni;
    auto nj   = cfg.nj;
    auto iter = 0;
    auto t    = 0.0;
    auto dx   = 1.0 / ni;
    auto dy   = 1.0 / nj;
    auto dt   = std::min(dx, dy) * 0.125;
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
    write_database(database);

    return 0;
}




// ============================================================================
int main(int argc, const char* argv[])
{
    std::set_terminate(Debug::terminate_with_backtrace);
    return main_2d(argc, argv);
}
