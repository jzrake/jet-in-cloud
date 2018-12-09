#include <fstream>
#include <iostream>
#include <map>




// ============================================================================
namespace nd {
    template<typename Writeable>
    void tofile(const Writeable& writeable, const std::string& fname)
    {
        std::ofstream outfile(fname, std::ofstream::binary | std::ios::out);
    
        if (! outfile.is_open())
        {
            throw std::invalid_argument("file " + fname + " could not be opened for writing");
        }
        auto s = writeable.dumps();
        outfile.write(s.data(), s.size());
        outfile.close();
    }
}




// ============================================================================
template <class T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& arr)
{
    std::copy(arr.cbegin(), arr.cend(), std::ostream_iterator<T>(o, " "));
    return o;
}




// ============================================================================
namespace cmdline 
{
    std::map<std::string, std::string> parse_keyval(int argc, const char* argv[]);

    template <typename T>
    inline void set_from_string(std::string source, T& value);

    template <>
    inline void set_from_string<std::string>(std::string source, std::string& value)
    {
        value = source;
    }

    template <>
    inline void set_from_string<int>(std::string source, int& value)
    {
        value = std::stoi(source);
    }

    template <>
    inline void set_from_string<double>(std::string source, double& value)
    {
        value = std::stod(source);
    }
}




// ============================================================================
namespace filesystem
{
    std::vector<std::string> split(std::string path);
    std::string join(std::vector<std::string> parts);
    std::string extension(std::string path);
    std::string parent(std::string path);
    void require_dir(std::string path);
    int remove_recurse(std::string path);
}




// ============================================================================
class Timer
{
public:
    Timer() : instantiated(std::clock())
    {
    }
    double seconds() const
    {
        return double (std::clock() - instantiated) / CLOCKS_PER_SEC;
    }
private:
    std::clock_t instantiated;
};




// ============================================================================
namespace debug
{
    void backtrace();
    void terminate_with_backtrace();
};
