#include <vector>
#include <string>
#include <cmath>
#include "app_utils.hpp"
#include "patches.hpp"
#include "jic.hpp"
#include "physics.hpp"
#include "ufunc.hpp"
#include "dbser.hpp"

namespace hydro = sru_hydro;
using namespace patches2d;




// ============================================================================
int main(int argc, const char* argv[])
{
    if (argc == 1)
    {
        std::cout << "usage: analyze chkpt.0000\n";
        return 0;
    }

    try {
        std::cout << argv[1] << " " << filesystem::join({argv[1], "config.json"}) << std::endl;

        auto cfg_fname = filesystem::join({argv[1], "config.json"});
        auto sts_fname = filesystem::join({argv[1], "status.json"});
        auto cfg_ifs = std::ifstream(cfg_fname);
        auto sts_ifs = std::ifstream(sts_fname);

        if (! cfg_ifs.is_open())
        {
            throw std::runtime_error("could not open config file: " + cfg_fname);
        }
        if (! sts_ifs.is_open())
        {
            throw std::runtime_error("could not open status file: " + sts_fname);
        }


        // Load and print the config, status, and database files
        // --------------------------------------------------------------------
        auto cfg = jic::run_config::from_json(cfg_ifs);
        auto sts = jic::run_status::from_json(sts_ifs);
        auto database = Database::load(FileSystemSerializer(argv[1], "r"));
        cfg.print     (std::cout);
        sts.print     (std::cout);
        database.print(std::cout);


        // Assemble the conserved variable array and convert it to primitives
        // --------------------------------------------------------------------
        auto cons_to_prim = ufunc::vfrom(hydro::cons_to_prim());
        auto X = database.assemble(0, cfg.num_blocks, 0, 1, 0, Field::cell_coords);
        auto U = database.assemble(0, cfg.num_blocks, 0, 1, 0, Field::conserved);
        auto P = cons_to_prim(U, X);

        for (int j = 0; j < P.shape(1); ++j)
        {
            auto fname = "prim-" + std::string(4 - std::to_string(j).length(), '0') + std::to_string(j) + ".dat";
            auto out = std::ofstream(fname);

            std::cout << "write " << fname << std::endl;

            for (int i = 0; i < P.shape(0); ++i)
            {
                const double dg = P(i, j, 0);
                const double vr = P(i, j, 1);
                const double vq = P(i, j, 2);
                const double vp = P(i, j, 3);
                const double pg = P(i, j, 4);
                const double Hg = dg + 4 * pg;
                const double vv = vr * vr + vq * vq + vp * vp;
                const double Gm = 1.0 / std::sqrt(1 - vv);
                const double ur = Gm * vr;
                const double Fr = (Hg * Gm * Gm - dg * Gm) * vr;
                out << X(i, j, 0) << " " << dg << " " << ur << " " << pg << " " << Fr << std::endl;
            }
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "\nERROR: ";
        std::cerr << e.what() << "\n\n";
        return 1;
    }
    return 0;
}
