#include <gtest/gtest.h>
#include <DataLoader/CubeLUT.hpp>

TEST(LUTParserTest, testBasicLUT)
{
    CubeLUT lut;
    EXPECT_EQ(lut.getType(), LUTType::UNKNOWN);
}
