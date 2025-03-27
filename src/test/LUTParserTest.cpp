#include <gtest/gtest.h>
#include <FileIO/CubeLUT.hpp>
#include <fstream>
#include <filesystem>
#include <variant>
#include <Eigen/Dense>
#include <CubeLUTMock.hpp>
#include <array>

constexpr float ERROR_DELTA = 0.0000001f;
constexpr int TABLE1D_INDEX = 0;
constexpr int TABLE3D_INDEX = 1;

using namespace ::testing;

class LUTParserTest : public ::testing::Test {};

TEST_F(LUTParserTest, SLOW_testTypical1DLUT)
{
    constexpr auto lutPath = "resources/alog.cube";
    CubeLUT lut;
    std::ifstream lutStream(lutPath);
    lut.loadCubeFile(lutStream);
    ASSERT_EQ(lut.getType(), LUTType::LUT1D);
    EXPECT_EQ(lut.getTable().index(), TABLE1D_INDEX);
    const auto& table = std::get<Table1D>(lut.getTable());
    EXPECT_EQ(table.size(), 4096 * 3);
    EXPECT_EQ(table.dimensions().size(), 2);
    EXPECT_EQ(table.dimension(0), 4096);
    EXPECT_EQ(table.dimension(1), 3);

    // value sanity check
    EXPECT_NEAR(table(0, 0), 0.011447058f, ERROR_DELTA);
    EXPECT_NEAR(table(2048, 1), 0.488697845f, ERROR_DELTA);
    EXPECT_NEAR(table(4095, 2), 0.980259102f, ERROR_DELTA);
}

TEST_F(LUTParserTest, SLOW_testTypical3DLUT)
{
    constexpr auto lutPath = "resources/xt3_flog_bt709.cube";
    CubeLUT lut;
    std::ifstream lutStream(lutPath);
    lut.loadCubeFile(lutStream);
    ASSERT_EQ(lut.getType(), LUTType::LUT3D);
    EXPECT_EQ(lut.getTable().index(), TABLE3D_INDEX);
    const auto& table = std::get<Table3D>(lut.getTable());
    EXPECT_EQ(table.size(), 33 * 33 * 33 * 3);
    EXPECT_EQ(table.dimensions().size(), 4);
    EXPECT_EQ(table.dimension(0), 33);
    EXPECT_EQ(table.dimension(1), 33);
    EXPECT_EQ(table.dimension(2), 33);
    EXPECT_EQ(table.dimension(3), 3);

    // value sanity check
    EXPECT_NEAR(table(0, 0, 0, 2), 0.0f, ERROR_DELTA);
    EXPECT_NEAR(table(16, 16, 16, 1), 0.500031f, ERROR_DELTA);
    EXPECT_NEAR(table(32, 32, 32, 2), 1.0f, ERROR_DELTA);
}

void testMinimal1DLUT(const Table1D& table) {
    EXPECT_EQ(table.size(), 2 * 3);
    EXPECT_EQ(table.dimensions().size(), 2);
    EXPECT_EQ(table.dimension(0), 2);
    EXPECT_EQ(table.dimension(1), 3);
}

TEST_F(LUTParserTest, testCRLF)
{
    constexpr auto exampleLUT = "TITLE \"Some LUT\"\r\nLUT_1D_SIZE 2\r\n\r\n0.1 0.2 0.3\r\n0.4 0.5 0.6\r\n";
    CubeLUT lut;
    std::istringstream lutStream(exampleLUT);
    lut.loadCubeFile(lutStream);
    ASSERT_EQ(lut.getType(), LUTType::LUT1D);
    EXPECT_EQ(lut.getTable().index(), TABLE1D_INDEX);
    const auto& table = std::get<Table1D>(lut.getTable());
    testMinimal1DLUT(table);

    // value sanity check
    EXPECT_NEAR(table(0, 0), 0.1f, ERROR_DELTA);
    EXPECT_NEAR(table(1, 2), 0.6f, ERROR_DELTA);
}

TEST_F(LUTParserTest, trailingWhitespaceInHeader) {
    std::istringstream lutStream;
    CubeLUTMock lut;
    EXPECT_CALL(lut, parseLUTTable);
    constexpr auto headerWithUnknownTags = \
        "#Some Sample Comment \n"
        "LUT_3D_SIZE 2 \n"
        " \n"
        "\t\r\n"
        "1.0 2.0 3.0\n";
    lutStream = std::istringstream(headerWithUnknownTags);
    EXPECT_NO_THROW(lut.loadCubeFile(lutStream));
    EXPECT_EQ(lut.getType(), LUTType::LUT3D);
}

namespace ValueClippingTestData {
    constexpr uint32_t VALUES_OUTSIDE_OF_RANGE_1D_LUT = 2u;
    constexpr auto minimal1DLUT = \
    "TITLE \"Some LUT\"\nLUT_1D_SIZE 2\n1.0 1.00001 0.3\n0.4 -0.52 0.6\n";

    constexpr uint32_t VALUES_OUTSIDE_OF_RANGE_3D_LUT = 5u;
    constexpr auto minimal3DLUT = \
    "TITLE \"Some LUT\"\nLUT_3D_SIZE 2\n"
    "1.1 100.0 -1.5\n"
    "0.4 0.5 0.6\n"
    "0.1 0.2 0.3\n"
    "0.4 0.5 0.6\n"
    "0.1 -0.00001 0.3\n"
    "0.4 0.5 0.6\n"
    "0.1 0.2 0.3\n"
    "0.4 0.5 1.000001\n";
} ;

struct ValueClippingTest : public ::testing::TestWithParam<std::pair<const char*, uint32_t>> {};

TEST_P(ValueClippingTest, clippingTest) {
    const auto [lutContent, lutValues] =  GetParam();
    CubeLUTMock lut;
    EXPECT_CALL(lut, clipValue).Times(lutValues);
    EXPECT_CALL(lut, parseLUTTable).WillOnce([&lut](std::istream& infile){ return lut.callBaseParseLUTTable(infile); });
    std::istringstream lutStream(lutContent);
    lut.loadCubeFile(lutStream);
    EXPECT_TRUE(lut.isDomainViolationDetected());
}

INSTANTIATE_TEST_SUITE_P(
        LUTParserTest,
        ValueClippingTest,
        ::testing::Values(
            std::make_pair(ValueClippingTestData::minimal1DLUT, ValueClippingTestData::VALUES_OUTSIDE_OF_RANGE_1D_LUT),
            std::make_pair(ValueClippingTestData::minimal3DLUT, ValueClippingTestData::VALUES_OUTSIDE_OF_RANGE_3D_LUT)
        )
);

std::string createHeaderWithSize(uint32_t lutSize, LUTType type) {
    if (type == LUTType::LUT1D) {
        return "TITLE \"Some LUT\"\nLUT_1D_SIZE " + std::to_string(lutSize) + "\n";
    }
    else {
        return "TITLE \"Some LUT\"\nLUT_3D_SIZE " + std::to_string(lutSize) + "\n";
    }
}

struct UnsupportedLUTSizeTest : public ::testing::TestWithParam<std::string> {};

constexpr uint32_t LUT_1D_MAX_SIZE = 65536;
constexpr uint32_t LUT_1D_MIN_SIZE = 2;
constexpr uint32_t LUT_3D_MAX_SIZE = 256;
constexpr uint32_t LUT_3D_MIN_SIZE = 2;

TEST_P(UnsupportedLUTSizeTest, lutSizeUnsupported) {
    const auto& lutContent = GetParam();
    CubeLUT lut;
    std::istringstream lutStream(lutContent);
    EXPECT_THROW(lut.loadCubeFile(lutStream), std::runtime_error);
    EXPECT_EQ(lut.getType(), LUTType::UNKNOWN);
}

INSTANTIATE_TEST_SUITE_P(
        LUTParserTest,
        UnsupportedLUTSizeTest,
        ::testing::Values(
            createHeaderWithSize(LUT_1D_MAX_SIZE + 1, LUTType::LUT1D),
            createHeaderWithSize(LUT_1D_MIN_SIZE - 1, LUTType::LUT1D),
            createHeaderWithSize(LUT_3D_MAX_SIZE + 1, LUTType::LUT3D),
            createHeaderWithSize(LUT_3D_MIN_SIZE - 1, LUTType::LUT3D)
        )
);

TEST_F(LUTParserTest, unknownLUTType) {
    constexpr auto exampleLUT = "TITLE \"Some LUT\"\n0.1 0.2 0.3\n0.4 0.5 0.6\n";
    CubeLUTMock lut;
    EXPECT_CALL(lut, parseLUTTable).Times(0);
    std::istringstream lutStream(exampleLUT);
    try
    {
        lut.loadCubeFile(lutStream);
    }
    catch (const std::runtime_error& ex)
    {
        EXPECT_STREQ("Unknown LUT type: specify the LUT_1D_SIZE/LUT_3D_SIZE tag", ex.what());
    }
    EXPECT_EQ(lut.getType(), LUTType::UNKNOWN);
}

TEST_F(LUTParserTest, reversedDomainBounds) {
    constexpr auto exampleLUT = "TITLE \"Some LUT\"\nLUT_1D_SIZE 2\nDOMAIN_MIN 1.0 1.0 1.0\nDOMAIN_MAX 0.0 0.0 0.0";
    CubeLUTMock lut;
    EXPECT_CALL(lut, parseLUTTable).Times(0);
    std::istringstream lutStream(exampleLUT);
    try
    {
        lut.loadCubeFile(lutStream);
    }
    catch (const std::runtime_error& ex)
    {
        EXPECT_STREQ("Domain bounds are reversed (DOMAIN_MIN is larger than DOMAIN_MAX)", ex.what());
    }
}

TEST_F(LUTParserTest, basicPropertiesTest) {
    std::istringstream lutStream;
    CubeLUTMock lut;
    EXPECT_CALL(lut, parseLUTTable);
    constexpr auto example1DLUT = \
        "TITLE \"Some LUT\"\n"
        "LUT_1D_SIZE 2\n"
        "DOMAIN_MIN -1.0 -2.0 -3.0\n"
        "DOMAIN_MAX 1.0 2.0 3.0";
    lutStream = std::istringstream(example1DLUT);
    lut.loadCubeFile(lutStream);
    EXPECT_THAT(lut.getDomainMin(), ElementsAre(-1.0, -2.0, -3.0));
    EXPECT_THAT(lut.getDomainMax(), ElementsAre(1.0, 2.0, 3.0));
    EXPECT_EQ(lut.getTitle(), "Some LUT");

    constexpr auto example3DLUT = \
        "TITLE \"Some Other LUT\"\n"
        "LUT_3D_SIZE 2\n"
        "DOMAIN_MIN 0.0 0.0 0.0\n"
        "DOMAIN_MAX 1.0 1.0 1.0";
    lutStream = std::istringstream(example3DLUT);
    EXPECT_CALL(lut, parseLUTTable);
    lut.loadCubeFile(lutStream);
    EXPECT_THAT(lut.getDomainMin(), ElementsAre(0.0, 0.0, 0.0));
    EXPECT_THAT(lut.getDomainMax(), ElementsAre(1.0, 1.0, 1.0));
    EXPECT_EQ(lut.getTitle(), "Some Other LUT");
}

TEST_F(LUTParserTest, unknownParamTest) {
    // unknown params should be ignored and file should be parsed correctly
    std::istringstream lutStream;
    CubeLUTMock lut;
    EXPECT_CALL(lut, parseLUTTable);
    constexpr auto lutWithUnknownTags = \
        "TITLE \"Some LUT\"\n"
        "LUT_1D_SIZE 2\n"
        "SOME_UNKNOWN_TAG 1000 20\n"
        "DOMAIN_MIN -1.0 -2.0 -3.0\n"
        "ANOTHER_ONE\n"
        "DOMAIN_MAX 1.0 2.0 3.0";
    lutStream = std::istringstream(lutWithUnknownTags);
    lut.loadCubeFile(lutStream);
    EXPECT_EQ(lut.getType(), LUTType::LUT1D);
    EXPECT_THAT(lut.getDomainMin(), ElementsAre(-1.0, -2.0, -3.0));
    EXPECT_THAT(lut.getDomainMax(), ElementsAre(1.0, 2.0, 3.0));
}
