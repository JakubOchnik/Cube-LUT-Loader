#include "MultidimDataUtils.hpp"

template <typename T>
std::vector<T> flatten4D(const std::vector < std::vector<std::vector<std::vector<T>>>>& orig)
{
	std::vector<int> out;
	for (const auto& zp : orig)
	{
		for (const auto& z : zp)
		{
			for (const auto& y : z) {
				out.insert(out.end(), y.begin(), y.end());
			}
		}
	}
	return out;
}

template <typename T>
std::vector<T> flatten3D(const std::vector < std::vector<std::vector<T>>>& orig)
{
	std::vector<int> out;
	for (const auto& z : orig)
	{
		for (const auto& y : z)
		{
			out.insert(out.end(), y.begin(), y.end());
		}
	}
	return out;
}

template <typename T>
std::vector<T> flatten2D(const std::vector < std::vector<T>>& orig)
{
	std::vector<int> out;
	for (const auto& y : z)
	{
		out.insert(out.end(), y.begin(), y.end());
	}
	return out;
}