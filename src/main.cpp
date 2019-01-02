#include <iostream>
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
#include "dbser.hpp"
#include "ufunc.hpp"
#include "jic.hpp"
#include "thread_pool.hpp"

using namespace patches2d;
namespace hydro = sru_hydro;
using run_config = jic::run_config;
using run_status = jic::run_status;




// ============================================================================
template <typename T>
std::vector<char> swap_bytes(const std::vector<T>& buffer)
{
    auto res = std::vector<char>(buffer.size() * sizeof(T));

    for (std::size_t n = 0; n < buffer.size(); ++n)
    {
        const char* src = reinterpret_cast<const char*>(buffer.data()) + n * sizeof(T);
        char* dst = static_cast<char*>(res.data()) + n * sizeof(T);

        for (unsigned int b = 0; b < sizeof(T); ++b)
        {
            dst[sizeof(T) - b - 1] = src[b];
        }
    }
    return res;
}

template <typename T>
void write_swapped_bytes_and_clear(std::ostream& os, std::vector<T>& buffer)
{
    auto bytes = swap_bytes(buffer);
    os.write(bytes.data(), bytes.size());
    buffer.clear();
}




// ============================================================================
void write_chkpt(const Database& database, run_config cfg, run_status sts, int count)
{
    auto chkpt = cfg.make_filename_chkpt(count);
    std::cout << "write checkpoint " << chkpt << std::endl;

    FileSystemSerializer serializer(chkpt, "w");
    database.dump(serializer);

    // Write the run config and status to json
    // ------------------------------------------------------------------------
    auto cfg_stream = std::fstream(cfg.make_filename_config(count), std::ios::out);
    auto sts_stream = std::fstream(cfg.make_filename_status(count), std::ios::out);

    cfg.tojson(cfg_stream);
    sts.tojson(sts_stream);
}

void write_vtk(const Database& database, run_config cfg, run_status /*sts*/, int count)
{
    auto filename = cfg.make_filename_vtk(count);

    std::cout << "write VTK " << filename << std::endl;
    filesystem::require_dir(filesystem::parent(filename));

    auto stream = std::fstream(filename, std::ios::out);
    auto cons_to_prim = ufunc::vfrom(hydro::cons_to_prim());
    auto vert = database.assemble(0, cfg.num_blocks, 0, 1, 0, Field::vert_coords);
    auto buffer = std::vector<float>();


    // ------------------------------------------------------------------------
    // Write header
    // ------------------------------------------------------------------------
    stream << "# vtk DataFile Version 3.0\n";
    stream << "My Data" << "\n";
    stream << "BINARY\n";
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
            buffer.push_back(x);
            buffer.push_back(0.0);
            buffer.push_back(z);
        }
    }
    write_swapped_bytes_and_clear(stream, buffer);


    // ------------------------------------------------------------------------
    // Write primitive data
    // ------------------------------------------------------------------------
    auto cons = database.assemble(0, cfg.num_blocks, 0, 1, 0, Field::conserved);
    auto cell = database.assemble(0, cfg.num_blocks, 0, 1, 0, Field::cell_coords);
    auto prim = cons_to_prim(cons, cell);
    stream << "CELL_DATA " << prim.shape(0) * prim.shape(1) << "\n";

    stream << "SCALARS " << "density " << "FLOAT " << 1 << "\n";
    stream << "LOOKUP_TABLE default\n";

    for (int j = 0; j < prim.shape(1); ++j)
    {
        for (int i = 0; i < prim.shape(0); ++i)
        {
            buffer.push_back(prim(i, j, 0));
        }
    }
    write_swapped_bytes_and_clear(stream, buffer);

    stream << "SCALARS " << "radial_velocity " << "FLOAT " << 1 << "\n";
    stream << "LOOKUP_TABLE default\n";

    for (int j = 0; j < prim.shape(1); ++j)
    {
        for (int i = 0; i < prim.shape(0); ++i)
        {
            buffer.push_back(prim(i, j, 1));
        }
    }
    write_swapped_bytes_and_clear(stream, buffer);

    stream << "SCALARS " << "pressure " << "FLOAT " << 1 << "\n";
    stream << "LOOKUP_TABLE default\n";

    for (int j = 0; j < prim.shape(1); ++j)
    {
        for (int i = 0; i < prim.shape(0); ++i)
        {
            buffer.push_back(prim(i, j, 4));
        }
    }
    write_swapped_bytes_and_clear(stream, buffer);
}




// ============================================================================
struct MeshGeometry
{
    MeshGeometry(
        nd::array<double, 3> A,
        const nd::array<double, 3>& B,
        const nd::array<double, 3>& C,
        const nd::array<double, 3>& D,
        const nd::array<double, 3>& E)
    : centroids_extended(A)
    , centroids(B)
    , volumes(C)
    , face_areas_i(D)
    , face_areas_j(E)
    {
    }

    nd::array<double, 3> centroids_extended;
    const nd::array<double, 3>& centroids;
    const nd::array<double, 3>& volumes;
    const nd::array<double, 3>& face_areas_i;
    const nd::array<double, 3>& face_areas_j;
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
auto advance_2d(nd::array<double, 3> U0, const MeshGeometry& G, double dt)
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
    auto evaluate_src = ufunc::vfrom(hydro::sph_geom_src_terms());
    auto cons_to_prim = ufunc::vfrom(hydro::cons_to_prim());
    auto godunov_flux_i = ufunc::vfrom(hydro::riemann_hlle({1, 0, 0}));
    auto godunov_flux_j = ufunc::vfrom(hydro::riemann_hlle({0, 1, 0}));
    auto extrap_l = ufunc::from([] (double a, double b) { return a - b * 0.5; });
    auto extrap_r = ufunc::from([] (double a, double b) { return a + b * 0.5; });
    auto flux_times_area = ufunc::vfrom(flux_times_area_formula);

    auto mi = U0.shape(0);
    auto mj = U0.shape(1);
    auto P0 = cons_to_prim(U0, G.centroids_extended);

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
void update_2d_threaded(ThreadPool& pool, Database& database, double dt, double rk_factor)
{
    using Result = std::pair<Database::Index, Database::Array>;
    auto futures = std::vector<std::future<Result>>();

    auto field = [] (Database::Index index, Field field)
    {
        std::get<3>(index) = field;
        return index;
    };

    auto update_task = [] (Database::Index index, const Database::Array& U,
			   const MeshGeometry& G, double dt)
    {
        return std::make_pair(index, advance_2d(U, G, dt));
    };

    for (const auto& patch : database.all(Field::conserved))
    {
        auto U = database.fetch(patch.first, 2, 2, 0, 0);
        auto G = MeshGeometry(
            database.fetch(field(patch.first, Field::cell_coords), 2, 2, 0, 0),
            database.at(patch.first, Field::cell_coords),
            database.at(patch.first, Field::cell_volume),
            database.at(patch.first, Field::face_area_i),
            database.at(patch.first, Field::face_area_j));
        futures.push_back(pool.enqueue(update_task, patch.first, U, G, dt));
    }

    for (auto& future : futures)
    {
        auto result = future.get();
        database.commit(result.first, result.second, rk_factor);
    }
}

void update(ThreadPool& pool, Database& database, double dt, int rk)
{
    switch (rk)
    {
        case 1:
            update_2d_threaded(pool, database, dt, 0.0);
            break;
        case 2:
            update_2d_threaded(pool, database, dt, 0.0);
            update_2d_threaded(pool, database, dt, 0.5);
            break;
        default:
            throw std::invalid_argument("rk must be 1 or 2");
    }
}




// ============================================================================
struct atmosphere
{
    atmosphere(double density_index, double temperature)
    : density_index(density_index)
    , temperature(temperature)
    {
    }

    inline std::array<double, 5> operator()(std::array<double, 2> X) const
    {
        const double r = X[0];
        const double a = density_index;
        return std::array<double, 5>{std::pow(r, -a), 0.0, 0.0, 0.0, temperature * std::pow(r, -a)};
    }
    double density_index;
    double temperature;
};




// ============================================================================
struct explosion
{
    inline std::array<double, 5> operator()(std::array<double, 2> X) const
    {
        double d = X[0] < 2 ? 1.0 : 0.1;
        double p = X[0] < 2 ? 1.0 : 0.125;
        return std::array<double, 5>{d, 0.0, 0.0, 0.0, p};
    }
};




// ============================================================================
struct jet_boundary_value
{
    jet_boundary_value(run_config cfg) : cfg(cfg) {}

    nd::array<double, 3> operator()(
        Database::Index index,
        PatchBoundary edge,
        int /*depth*/,
        const nd::array<double, 3>& patch) const
    {
        if (std::get<3>(index) == Field::conserved)
        {
            switch (edge)
            {
                case PatchBoundary::il: return inflow_inner(patch);
                case PatchBoundary::ir: return zero_gradient_outer(patch);
                default: throw;
            }
        }
        else if (std::get<3>(index) == Field::cell_coords)
        {
            switch (edge)
            {
                case PatchBoundary::il: return zero_gradient_inner(patch);
                case PatchBoundary::ir: return zero_gradient_outer(patch);
                default: throw;
            }            
        }
        throw;
    }

    nd::array<double, 3> zero_gradient_inner(const nd::array<double, 3>& patch) const
    {
        auto _ = nd::axis::all();
        auto U = nd::array<double, 3>(2, patch.shape(1), patch.shape(2));
        U.select(0, _, _) = patch.select(0, _, _);
        U.select(1, _, _) = patch.select(0, _, _);
        return U;
    }

    nd::array<double, 3> zero_gradient_outer(const nd::array<double, 3>& patch) const
    {
        auto _ = nd::axis::all();
        auto U = nd::array<double, 3>(2, patch.shape(1), patch.shape(2));
        U.select(0, _, _) = patch.select(patch.shape(0) - 1, _, _);
        U.select(1, _, _) = patch.select(patch.shape(0) - 1, _, _);
        return U;
    }

    nd::array<double, 3> inflow_inner(const nd::array<double, 3>& patch) const
    {
        auto prim_to_cons = ufunc::vfrom(hydro::prim_to_cons());
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

    hydro::Vars jet_inlet(double q) const
    {
        auto q0 = 0.0;
        auto q1 = M_PI;
        auto u0 = cfg.jet_velocity;
        auto dg = cfg.jet_density;
        auto dq = cfg.jet_opening_angle;
        auto f0 = u0 * std::exp(-std::pow((q - q0) / dq, 2));
        auto f1 = u0 * std::exp(-std::pow((q - q1) / dq, 2));
        auto inflowP = hydro::Vars{dg, f0 + f1, 0.0, 0.0, cfg.temperature * dg};
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
            case PatchBoundary::il: return zero_gradient_inner(patch);
            case PatchBoundary::ir: return zero_gradient_outer(patch);
            case PatchBoundary::jl: return nd::array<double, 3>();
            case PatchBoundary::jr: return nd::array<double, 3>();
        }
	   throw;
    }

    nd::array<double, 3> zero_gradient_outer(const nd::array<double, 3>& patch) const
    {
        auto _ = nd::axis::all();
        auto U = nd::array<double, 3>(2, patch.shape(1), 5);
        U.select(0, _, _) = patch.select(patch.shape(0) - 1, _, _);
        U.select(1, _, _) = patch.select(patch.shape(0) - 1, _, _);
        return U;
    }

    nd::array<double, 3> zero_gradient_inner(const nd::array<double, 3>& patch) const
    {
        auto _ = nd::axis::all();
        auto U = nd::array<double, 3>(2, patch.shape(1), 5);
        U.select(0, _, _) = patch.select(0, _, _);
        U.select(1, _, _) = patch.select(0, _, _);
        return U;
    }
};




// ============================================================================
Database::BoundaryValue get_boundary_value(run_config cfg)
{
    if (cfg.test_mode)
    {
        return simple_boundary_value();
    }
    else
    {
        return jet_boundary_value(cfg);
    }
}

Database create_database(run_config cfg)
{
    if (! cfg.restart.empty())
    {
        FileSystemSerializer serializer(cfg.restart, "r");
        auto database = Database::load(serializer);
        database.set_boundary_value(get_boundary_value(cfg));
        return database;
    }

    auto target_radial_zone_count = cfg.nr * std::log10(cfg.outer_radius);
    auto block_size = target_radial_zone_count / cfg.num_blocks;

    auto ni = block_size;
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
    auto prim_to_cons = ufunc::vfrom(hydro::prim_to_cons());

    for (int i = 0; i < cfg.num_blocks; ++i)
    {
        double r0 = std::pow(cfg.outer_radius, double(i + 0) / cfg.num_blocks);
        double r1 = std::pow(cfg.outer_radius, double(i + 1) / cfg.num_blocks);

        auto x_verts = mesh_vertices(ni, nj, {r0, r1, 0, M_PI});
        auto x_cells = mesh_cell_centroids(x_verts);
        auto v_cells = mesh_cell_volumes(x_verts);
        auto a_faces_i = mesh_face_areas_i(x_verts);
        auto a_faces_j = mesh_face_areas_j(x_verts);

        database.insert(std::make_tuple(i, 0, 0, Field::vert_coords), x_verts);
        database.insert(std::make_tuple(i, 0, 0, Field::cell_coords), x_cells);
        database.insert(std::make_tuple(i, 0, 0, Field::cell_volume), v_cells);
        database.insert(std::make_tuple(i, 0, 0, Field::face_area_i), a_faces_i);
        database.insert(std::make_tuple(i, 0, 0, Field::face_area_j), a_faces_j);

        if (cfg.test_mode)
        {
            auto initial_data = ufunc::vfrom(explosion());
            database.insert(std::make_tuple(i, 0, 0, Field::conserved), prim_to_cons(initial_data(x_cells)));
            
        }
        else
        {
            auto initial_data = ufunc::vfrom(atmosphere(cfg.density_index, cfg.temperature));
            database.insert(std::make_tuple(i, 0, 0, Field::conserved), prim_to_cons(initial_data(x_cells)));
        }
    }

    database.set_boundary_value(get_boundary_value(cfg));
    return database;
}




// ============================================================================
Scheduler create_scheduler(run_config& cfg, run_status& sts, const Database& database)
{
    auto scheduler = Scheduler(sts.time);

    auto task_vtk = [&cfg, &sts, &database] (int count)
    {
        sts.vtk_count = count + 1;
        write_vtk(database, cfg, sts, count);
    };

    auto task_chkpt = [&cfg, &sts, &database] (int count)
    {
        sts.chkpt_count = count + 1;
        write_chkpt(database, cfg, sts, count);
    };

    scheduler.repeat("write vtk", cfg.vtki, sts.vtk_count, task_vtk);
    scheduler.repeat("write checkpoint", cfg.cpi, sts.chkpt_count, task_chkpt);

    return scheduler;
}




// ============================================================================
int run(int argc, const char* argv[])
{
    auto cfg = run_config::from_argv(argc, argv).validate();
    auto sts = run_status::from_config(cfg);
    auto database  = create_database(cfg);
    auto scheduler = create_scheduler(cfg, sts, database);
    auto dt = 0.25 * M_PI / cfg.nr;


    ThreadPool thread_pool(cfg.num_threads);


    // ========================================================================
    // Initial report
    // ========================================================================
    std::cout << "\n";
    cfg      .print(std::cout);
    sts      .print(std::cout);
    database .print(std::cout);
    scheduler.print(std::cout);

    std::cout << std::string(52, '=') << "\n";
    std::cout << "Main loop:\n\n";


    // ========================================================================
    // Main loop
    // ========================================================================
    while (sts.time < cfg.tfinal)
    {
        scheduler.dispatch(sts.time);

        auto timer = Timer();
        update(thread_pool, database, dt, cfg.rk);

        sts.time += dt;
        sts.iter += 1;
        sts.wall += timer.seconds();

        auto kzps = database.num_cells(Field::conserved) / 1e3 / timer.seconds();
        std::printf("[%04d] t=%3.3lf kzps=%3.2lf\n", sts.iter, sts.time, kzps);
        std::fflush(stdout);
    }
    scheduler.dispatch(sts.time);


    // ========================================================================
    // Final report
    // ========================================================================
    std::cout << "\n";
    std::cout << std::string(52, '=') << "\n";
    std::cout << "Run completed:\n\n";
    std::printf("\taverage kzps=%f\n", database.num_cells(Field::conserved) / 1e3 / sts.wall * sts.iter);
    std::cout << "\n";

    return 0;
}




// ============================================================================
int main(int argc, const char* argv[])
{
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
