#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>

#include "Common.h"

/** Singleton class for reading config.ini and storing values. 
 * Usage: 
 *      Get reference from anywhere: ConfigParser& config = ConfigParser::Get();
 *      Access a value [Section] key = ... : double value = GetDouble(sectionName, key, defaultValue = 0.)
*/
class COLLISION_API ConfigParser {

// Macro here gives this default visibility, otherwise we may get linker errors (we had these at least on one Apple clang system)

public:

    // Static singleton getter
    static ConfigParser& get() {
        static ConfigParser parserInstance;
        return parserInstance;
    }

    // Read a config file and store contents in our internal map. Return value is false if the read failed
    bool load(const std::string& filename) {

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "! Failed to open config file: " << filename << std::endl;
            return false;
        }

        std::string section, line;
        while (std::getline(file, line)) {
            parseLine(line, section);
        }

        file.close();
        return true;
    }


    std::string getString(const std::string& section, const std::string& key, const std::string& defaultValue = "") const {
        auto sectionIter = config.find(section);
        if (sectionIter != config.end()) {
            auto keyIter = sectionIter->second.find(key);
            if (keyIter != sectionIter->second.end()) {
                return keyIter->second;
            }
        }
        std::cerr << "Warning: Attempted to access config variable [" << section << "] " << key << ", but no such key was found\n"; 
        return defaultValue;
    }

    int getInt(const std::string& section, const std::string& key, int defaultValue = 0) const {
        std::string value = getString(section, key, "");
        if (!value.empty()) {
            try {
                return std::stoi(value);
            } catch (const std::invalid_argument& e) {
                // Handle conversion error
            }
        }
        return defaultValue;
    }
    

    double getDouble(const std::string& section, const std::string& key, double defaultValue = 0.0) const {
        std::string value = getString(section, key, "");
        if (!value.empty()) {
            try {
                // Support numbers in scientific notation
                std::istringstream ss(value);
                ss >> std::scientific >> defaultValue;
                return defaultValue;
            } catch (const std::invalid_argument& e) {
                // Handle conversion error
            }
        }
        return defaultValue;
    }


    bool getBool(const std::string& section, const std::string& key, bool defaultValue = false) const {
        std::string value = getString(section, key, "");
        if (!value.empty()) {
            if (value == "true" || value == "1") {
                return true;
            } else if (value == "false" || value == "0") {
                return false;
            }
        }
        return defaultValue;
    }


    void printContents() const {
        for (const auto& section : config) {
            std::cout << "[" << section.first << "]\n";
            for (const auto& pair : section.second) {
                std::cout << pair.first << " = " << pair.second << "\n";
            }
        } 
    }


private:

    // Make the default constructor private to prevent external instantiation
    ConfigParser() {}

    // Nested map. Ini files are of form [Section] key = value, each section gets its own map of key, value pairs
    std::map<std::string, std::map<std::string, std::string>> config;


    void parseLine(const std::string& line, std::string& currentSection) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            // Skip empty lines and comments
            return;
        } else if (trimmed[0] == '[' && trimmed[trimmed.size() - 1] == ']') {
            // Section header
            currentSection = trimmed.substr(1, trimmed.size() - 2);
        } else {
            // Key-value pair within a section
            size_t equalsPos = trimmed.find('=');
            if (equalsPos != std::string::npos) {
                std::string key = trim(trimmed.substr(0, equalsPos));
                std::string value = trim(trimmed.substr(equalsPos + 1));
                config[currentSection][key] = value;
            }
        }
    }


    // Get a trimmed version of a string
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        size_t last = str.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }
        return str.substr(first, (last - first + 1));
    }
};

