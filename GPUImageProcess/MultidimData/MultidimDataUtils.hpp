#pragma once
#include <vector>

/*
template <typename T>
std::vector<T> flatten4D(const std::vector < std::vector<std::vector<std::vector<T>>>>&);

template <typename T>
std::vector<T> flatten3D(const std::vector < std::vector<std::vector<T>>>&);

template <typename T>
std::vector<T> flatten2D(const std::vector < std::vector<T>>&);
*/

template <typename T>
std::vector<T> flatten4D(const std::vector < std::vector<std::vector<std::vector<T>>>>& orig)
{
	const int dim = orig[0].size();
	std::vector<T> out(0, pow(dim, 3) * 3);
	// r - w; g - z; b - y; kolor - x
	for (const auto& w : orig)
	{
		for (const auto& z : w)
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
	std::vector<T> out;
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
	std::vector<T> out;
	for (const auto& y : z)
	{
		out.insert(out.end(), y.begin(), y.end());
	}
	return out;
}