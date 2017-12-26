#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "bounding_box.hpp"

using bbox = ppc::bounding_box<int>;
using point = bbox::point_type;

TEST_CASE("Bounding box initialization", "[bbox]")
{
	point min{ 0, 0, 0 };
	point max{ 10, 10, 10 };
	point center{ 5, 5, 5 };
	bbox box{ min, max };

	REQUIRE(box.min == min);
	REQUIRE(box.max == max);
}

TEST_CASE("Expand a bounding box with a point", "[expand]")
{
	bbox box{ { 0, 0, 0 },{ 10, 10, 10 } };

	point toContain{ -1, 15, 5 };
	ppc::expand(box, toContain);

	REQUIRE(box.min == point{ -1, 0, 0 });
	REQUIRE(box.max == point{ 10, 15, 10 });
}

TEST_CASE("Expand a bounding box with another bounding box", "[expand]")
{
	bbox box1{ { 0, 0, 0 },{ 10, 10, 10 } };
	bbox box2{ { 5, -1, 5 },{ 15, 9, 12 } };

	ppc::expand(box1, box2);

	REQUIRE(box1.min == point{ 0, -1, 0 });
	REQUIRE(box1.max == point{ 15, 10, 12 });
}

TEST_CASE("Intersection between two bounding boxes", "[intersection]")
{
	bbox box{ { 0, 0, 0 },{ 10, 10, 10 } };
	bbox box1{ { 5, 5, 5 },{ 12, 13, 14 } };
	bbox box2{ { 2, 2, 2 },{ 7, 7, 7 } };
	bbox box3{ { 10, 10, 10 },{ 11, 12, 13 } };
	bbox box4{ { -5, -6, -7 },{ -1, -1, -1 } };

	REQUIRE(ppc::intersects(box, box1));
	REQUIRE(ppc::intersects(box, box2));
	REQUIRE(ppc::intersects(box, box3));
	REQUIRE(!ppc::intersects(box, box4));
}