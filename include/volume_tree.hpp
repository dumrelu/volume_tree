#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace ppc
{
	namespace detail
	{
		template <
			typename V,
			typename T,
			template <typename> typename Ptr
		>
		struct node
		{
			using volume_type = V;
			using mapped_type = T;
			using value_type = std::pair<volume_type, mapped_type>;
			using value_ptr = Ptr<value_type>;
			using node_ptr = Ptr<node>;

			value_ptr value{};
			node_ptr parent{};
			node_ptr left{};
			node_ptr right{};

			node(value_ptr val = nullptr)
				: value(val)
			{
			}
		};

		template <
			typename Value, 
			typename NodePtr
		>
		struct iterator
		{
			using value_type = Value;
			using const_value_type = const std::remove_cv_t<value_type>;
			using node_ptr = NodePtr;

			iterator(node_ptr current)
				: m_current{ current }
			{
				if (m_current)
				{
					for (; m_current->left != nullptr; m_current = m_current->left);
				}
			}

			bool operator==(const iterator& other) const
			{
				return m_current == other.m_current;
			}

			bool operator!=(const iterator& other) const
			{
				return !(*this == other);
			}

			value_type& operator*()
			{
				assert(m_current && m_current->value != nullptr);
				return *m_current->value;
			}

			const_value_type& operator*() const
			{
				assert(m_current && m_current->value != nullptr);
				return *m_current->value;
			}

			value_type* operator->()
			{
				assert(m_current && m_current->value != nullptr);
				return &(*m_current->value);
			}

			const_value_type* operator->() const
			{
				assert(m_current && m_current->value != nullptr);
				return &(*m_current->value);
			}

			iterator& operator++()
			{
				assert(m_current != nullptr);

				if (m_current->parent == nullptr)
				{
					m_current = nullptr;
				}
				else if (m_current->parent->left == m_current)
				{
					if (!m_current->parent->right)
					{
						m_current = m_current->parent;
						++(*this);
					}
					else
					{
						m_current = m_current->parent->right;
						for (; m_current->left != nullptr; m_current = m_current->left);
					}
				}
				else //if (m_current->parent->right == m_current)
				{
					m_current = m_current->parent;
					++(*this);
				}

				return *this;
			}

			iterator operator++(int)
			{
				auto tmp = *this;
				++(*this);
				return tmp;
			}

		private:
			node_ptr m_current{};
		};

		template <template <typename> typename Ptr, typename T>
		struct PtrType
		{
			using type = typename Ptr<T>::type;
		};
	}

	template <
		typename V, 
		typename T, 
		typename Compare, 
		typename Intersect, 
		typename Expand, 
		typename Allocator
	>
	class volume_tree
	{
	public:
		using allocator_type = Allocator;
		using volume_compare = Compare;
		using volume_intersection = Intersect;
		using volume_expand = Expand;
		template <typename P> using ptr_type = typename detail::PtrType<typename Allocator::ptr_type, P>::type;
		using node_type = detail::node<V, T, ptr_type>;
		using node_ptr = ptr_type<node_type>;
		using volume_type = typename node_type::volume_type;
		using mapped_type = typename node_type::mapped_type;
		using value_type = typename node_type::value_type;
		using iterator = detail::iterator<value_type, node_ptr>;
		using const_iterator = detail::iterator<const value_type, node_ptr>;
		using size_type = std::size_t;
		//TODO: pointers, references, etc

		iterator insert(const value_type& value)
		{
			auto valuePtr = m_allocator.allocate<value_type>(value);
			if (m_size == 0)
			{
				m_root = m_allocator.allocate<node_type>(valuePtr);
				++m_size;
				return iterator{ m_root };
			}
			else if (m_size == 1)
			{
				auto newRoot = m_allocator.allocate<node_type>();
				newRoot->left = m_root;
				newRoot->right = m_allocator.allocate<node_type>(valuePtr);
				m_root = newRoot;
				++m_size;
				return iterator{ m_root->right };
			}
			else
			{
				return iterator{nullptr};
			}
		}

		iterator begin() { return iterator{ m_root }; }
		const_iterator begin() const { return const_iterator{ m_root }; }
		const_iterator cbegin() const { return const_iterator{ m_root }; }

		iterator end() { return iterator{ nullptr }; }
		const_iterator end() const { return const_iterator{ nullptr }; }
		const_iterator cend() { return const_iterator{ nullptr }; }

		size_type size() const { return m_size; }

	private:
		node_ptr m_root;
		allocator_type m_allocator;
		volume_compare m_compare;
		volume_intersection m_intersects;
		volume_expand m_expand;
		size_type m_size{};
	};
}