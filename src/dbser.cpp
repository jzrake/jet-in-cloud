#include <map>
#include "dbser.hpp"
#include "json.hpp"
#include "app_utils.hpp"
using namespace patches2d;




// ============================================================================
FileSystemSerializer::FileSystemSerializer(std::string chkpt, std::string mode) : chkpt(chkpt)
{
    if (mode == "w")
    {
        filesystem::remove_recurse(chkpt);
        filesystem::require_dir(chkpt);
    }
    else if (mode == "r")
    {
    }
    else
    {
        throw std::invalid_argument("mode must be r or w");
    }
}

std::vector<std::string> FileSystemSerializer::list_fields(std::string patch_index) const
{
    return filesystem::listdir(filesystem::join({chkpt, patch_index}));
}

std::vector<std::string> FileSystemSerializer::list_patches() const
{
    auto res = std::vector<std::string>();

    for (auto item : filesystem::listdir(chkpt))
    {
        if (filesystem::isdir(filesystem::join({chkpt, item})))
        {
            res.push_back(item);
        }
    }
    return res;
}

nd::array<double, 3> FileSystemSerializer::read_array(std::string path) const
{
    std::cout << "reading array " << path << std::endl;
    auto ifs = std::ifstream(filesystem::join({chkpt, path}));
    auto str = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    return nd::array<double, 3>::loads(str);
}

std::array<int, 2> FileSystemSerializer::read_block_size() const
{
    auto bname = filesystem::join({chkpt, "block_size.json"});
    auto j = nlohmann::json();
    auto is = std::ifstream(bname);

    if (! is.is_open())
    {
        throw std::runtime_error("cannot read block size from " + bname);
    }
    is >> j;

    return {j["ni"], j["nj"]};
}

Database::Header FileSystemSerializer::read_header() const
{
    auto hname = filesystem::join({chkpt, "header.json"});
    auto j = nlohmann::json();
    auto is = std::ifstream(hname);

    if (! is.is_open())
    {
        throw std::runtime_error("cannot read header from " + hname);
    }
    is >> j;
    Database::Header header;

    for (auto field : j.items())
    {
        auto num = field.value()[0];
        auto loc = patches2d::parse_location(field.value()[1]);
        auto ind = patches2d::parse_field(field.key());
        header.emplace(ind, FieldDescriptor(num, loc));
    }
    return header;
}

void FileSystemSerializer::write_array(std::string path, const nd::array<double, 3>& patch) const
{
    auto fname = filesystem::join({chkpt, path});
    filesystem::require_dir(filesystem::parent(fname));
    nd::tofile(patch, fname);
}

void FileSystemSerializer::write_header(Database::Header header) const
{
    auto j = nlohmann::json();
    auto os = std::ofstream(filesystem::join({chkpt, "header.json"}));

    for (auto item : header)
    {
        j[to_string(item.first)] = {item.second.num_fields, to_string(item.second.location)};
    }
    os << std::setw(4) << j << std::endl;
}

void FileSystemSerializer::write_block_size(std::array<int, 2> block_size) const
{
    auto j = nlohmann::json();
    j["ni"] = block_size[0];
    j["nj"] = block_size[1];
    auto os = std::ofstream(filesystem::join({chkpt, "block_size.json"}));
    os << std::setw(4) << j << std::endl;
}
