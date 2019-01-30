#include <ostream>
#include <string>
#include <map>
#include "visit_struct.hpp"




// ============================================================================
namespace jic {
    struct run_status;
    struct run_config;
}




// ============================================================================
struct jic::run_status
{
    static run_status from_file(std::string restart);
    static run_status from_json(std::istream& is);
    static run_status from_config(const run_config& cfg);

    void print(std::ostream& os) const;
    void tojson(std::ostream& os) const;

    template<typename Callable>
    void foreach(Callable f)
    {
        visit_struct::for_each(*this, [f] (std::string name, auto& value)
        {
            f(name, value);
        });
    }

    /** Data members */
    double time     = 0.0;
    double wall     = 0.0;
    int iter        = 0;
    int vtk_count   = 0;
    int chkpt_count = 0;
};




// ============================================================================
struct jic::run_config
{
    static run_config from_json(std::istream& is);
    static run_config from_argv(int argc, const char* argv[]);

    void print(std::ostream& os) const;
    void tojson(std::ostream& os) const;
    run_config validate() const;
    std::string make_filename_chkpt(int count) const;
    std::string make_filename_vtk(int count) const;
    std::string make_filename_status(int count) const;
    std::string make_filename_config(int count) const;

    template<typename Callable>
    void foreach(Callable f)
    {
        visit_struct::for_each(*this, [f] (std::string name, auto& value)
        {
            f(name, value);
        });
    }

    /** Run control */
    std::string outdir  = "data";
    std::string restart = "";
    double tfinal       = 0.1;
    double cpi          = 1.0;
    double vtki         = 0.0;
    int rk              = 1;
    int nr              = 32;
    int num_threads     = 1;
    int num_blocks      = 12;
    int test_mode       = 0;

    /** Physics setup */
    double jet_opening_angle = 0.5;
    double jet_velocity      = 0.9;
    double jet_density       = 0.01;
    double density_index     = 2.0;
    double temperature       = 0.01;
    double outer_radius      = 10.0;
};
