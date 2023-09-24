#include <gmock/gmock.h>
#include <DataLoader/CubeLUT.hpp>
#include <string>
#include <array>

class CubeLUTMock : public CubeLUT {
public:    
    MOCK_METHOD(float, clipValue, (float, int), (override, const));
    MOCK_METHOD(void, parseLUTTable, (std::istream& infile), (override));
    
    bool isDomainViolationDetected() {
        return domainViolationDetected;
    }

    void callBaseParseLUTTable(std::istream& infile) {
        return CubeLUT::parseLUTTable(infile);
    }

    std::string getTitle() {
        return title;
    }

    std::array<float, 3> getDomainMin() {
        return domainMin;
    }

    std::array<float, 3> getDomainMax() {
        return domainMax;
    }
};
