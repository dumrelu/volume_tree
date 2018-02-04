#pragma once

#include "box_tree_interface.hpp"
#include "volume_tree.hpp"
#include "host_allocator.hpp"

namespace ppc
{
	namespace box_tree
	{
		struct cpu_box_tree : public box_tree_interface
		{
			virtual std::string get_name() const override
			{
				return "CPU box tree";
			}

			virtual void insert(const box_value_pair& value) override
			{
				m_tree.insert(value);
			}

			virtual void insert(const box_value_pairs& values) override
			{
				for (const auto& value : values)
				{
					m_tree.insert(value);
				}
			}

			virtual std::size_t size() const override
			{
				return static_cast<int>(m_tree.size());
			}

			virtual values find(const bboxes& bboxes) const override
			{
				values vals;
				for (const auto& box : bboxes)
				{
					auto it = m_tree.find(box);
					if (it != m_tree.end())
					{
						vals.push_back(it->second);
					}
				}
				return vals;
			}

		private:
			using bbox_tree = ppc::volume_tree<
				bbox,
				int,
				ppc::Intersect,
				ppc::Expand,
				std::less<bbox>,
				ppc::host_allocator
			>;

			bbox_tree m_tree;
		};
	}
}