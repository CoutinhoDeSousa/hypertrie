#ifndef HYPERTRIE_EINSUM_HPP
#define HYPERTRIE_EINSUM_HPP

#include <utility>
#include <tsl/hopscotch_set.h>

#include "Dice/einsum/internal/Operator.hpp"
#include "Dice/einsum/internal/CartesianOperator.hpp"
#include "Dice/einsum/internal/JoinOperator.hpp"
#include "Dice/einsum/internal/ResolveOperator.hpp"
#include "Dice/einsum/internal/CountOperator.hpp"
#include "Dice/einsum/internal/EntryGeneratorOperator.hpp"
#include "Dice/einsum/internal/Context.hpp"

extern "C"
{

#include <igraph/igraph.h>
#include <igraph/igraph_cliques.h>  // import for 3.3 igraph_maximal_independent_vertex_sets
}
namespace einsum::internal {

	template<typename value_type, typename key_part_type, template<typename, typename> class map_type,
			template<typename> class set_type>
	std::shared_ptr<Operator<value_type, key_part_type, map_type, set_type>>
	Operator<value_type, key_part_type, map_type, set_type>::construct(const std::shared_ptr<Subscript> &subscript,
																	   const std::shared_ptr<Context> &context) {
	    // hier an der stelle wird gehooked d.h
		switch (subscript->type) {
			case Subscript::Type::Join: // std fall
				return std::make_shared<JoinOperator<value_type, key_part_type, map_type, set_type>>(subscript,
																									 context);
			case Subscript::Type::Resolve: // ergebnisse vorkommen sollen
				return std::make_shared<ResolveOperator<value_type, key_part_type, map_type, set_type>>(subscript,
																										context);
			case Subscript::Type::Count: // // lonely auflösen , dh wenn sie nicht vorkommen soll
				return std::make_shared<CountOperator<value_type, key_part_type, map_type, set_type>>(subscript,
																									  context);
			case Subscript::Type::Cartesian: //
				return std::make_shared<CartesianOperator<value_type, key_part_type, map_type, set_type>>(subscript,
																										  context);
			case Subscript::Type::EntryGenerator: //
				return std::make_shared<EntryGeneratorOperator<value_type, key_part_type, map_type, set_type>>(
						subscript, context);
			default:
				throw std::invalid_argument{"subscript is of an undefined type."};
		}
	}

	template<typename value_type, typename key_part_type, template<typename, typename> class map_type,
			template<typename> class set_type>
	class Einsum {

		using const_BoolHypertrie_t = const_BoolHypertrie<key_part_type, map_type, set_type>;
		using Join_t = Join<key_part_type, map_type, set_type>;
		using Operator_t = Operator<value_type, key_part_type, map_type, set_type>;
		using Entry_t = Entry<key_part_type, value_type>;
		using Key_t = typename Entry_t::key_type;


		std::shared_ptr<Subscript> subscript{};
		std::shared_ptr<Context> context{};
		std::vector<const_BoolHypertrie_t> operands{};
		std::shared_ptr<Operator_t> op{};
		Entry_t entry{};


	public:
		Einsum() = default;

		Einsum(std::shared_ptr<Subscript> subscript, const std::vector<const_BoolHypertrie_t> &operands,
			   TimePoint timeout = TimePoint::max())
				: subscript(std::move(subscript)), context{std::make_shared<Context>(timeout)},
				  operands(operands),
				  op{Operator_t::construct(this->subscript, context)},
				  entry{0, Key_t(this->subscript->resultLabelCount(), std::numeric_limits<key_part_type>::max())} {

            igraph_t graph;
            igraph_vector_t v;
            igraph_vector_ptr_t mis = IGRAPH_VECTOR_PTR_NULL;
            igraph_vector_ptr_init(&mis, 0);
            /*
             * Start building a graph
             */
            std::vector<int> edges;

            for (const auto &operand_sc : this->subscript->getRawSubscript().operands) {
                //das noch verbessern
                if(operand_sc.size() == 2){
                    edges.push_back(operand_sc[0]);
                    edges.push_back(operand_sc[1]);
                }
                else if(operand_sc.size() == 3){
                    edges.push_back(operand_sc[0]);
                    edges.push_back(operand_sc[1]);
                    edges.push_back(operand_sc[0]);
                    edges.push_back(operand_sc[2]);
                    edges.push_back(operand_sc[1]);
                    edges.push_back(operand_sc[2]);
                }
            }

            // hier evtl in real_t array convertieren

            int sizeArray = edges.size();
            igraph_real_t e[sizeArray];
            std::copy(edges.begin(),edges.end(),e);
            igraph_vector_view(&v, e, sizeof(e)/sizeof(double));
            igraph_create(&graph,&v, 0 , IGRAPH_UNDIRECTED);

            igraph_maximal_independent_vertex_sets(&graph,&mis);

            /* // find the "best" mis
            size_t best_mis = 0;
            size_t best_size = 0;

            // bestes auswählen anhat der weights (die ich noch nciht habe, aufzaehlen in der for oben
            const size_t set_number_count = igraph_vector_ptr_size(&mis);
            for (size_t i = 0 ; i < set_number_count; i++){
                igraph_vector_t *tmp_mis = (igraph_vector_t*) igraph_vector_ptr_e(&mis,long(i));

                size_t tmp_mis_size = igraph_vector_size(tmp_mis);
                size_t x = 0;
                for (size_t j = 0 ; j < tmp_mis_size; j++){
                    x += r_wei
                }

            }
             */

            igraph_vector_t *best_mis = (igraph_vector_t*)igraph_vector_ptr_e(&mis, long(0)); // 0 ersetzen durch "bestes"
            const size_t size_best_mis = igraph_vector_size(best_mis);
            std::vector<size_t> vmis;
            try{
                vmis.resize(size_best_mis);
                for(size_t j = 0 ; j < size_best_mis; j++){
                    vmis[j] += size_t(igraph_vector_e(best_mis, long(j)));
                }
            }
            catch (std::bad_alloc &error) {
                igraph_destroy(&graph);
                igraph_vector_ptr_destroy(&mis);
                throw error;
            }

            igraph_destroy(&graph);
            igraph_vector_ptr_destroy(&mis);

            /*
             * End building a graph
             */



            // set Operand reihenfolge
		}

		[[nodiscard]] const std::shared_ptr<Subscript> &getSubscript() const {
			return subscript;
		}

		const std::vector<const_BoolHypertrie_t> &getOperands() const {
			return operands;
		}

		const std::shared_ptr<Operator_t> &getOp() const {
			return op;
		}

		struct iterator {
		private:

			std::shared_ptr<Operator_t> op;
			Entry_t *current_entry;
			bool ended_ = false;
		public:
			iterator() = default;

			explicit iterator(Einsum &einsum, Entry_t &entry) : op(einsum.op), current_entry{&entry} {}

			iterator &operator++() {
				op->next();
				return *this;
			}

			inline const Entry<key_part_type, value_type> &operator*() {
				return *current_entry;
			}

			inline const Entry<key_part_type, value_type> &value() {
				return *current_entry;
			}

			operator bool() const {
				return not op->ended();

			}

			[[nodiscard]] inline bool ended() const { return op->ended(); }

		};

		iterator begin() {
			op->load(operands, entry);
			return iterator{*this, entry};
		}

		[[nodiscard]] bool end() const {
			return false;
		}

		void clear() {
		}
	};

	template<typename key_part_type, template<typename, typename> class map_type,
			template<typename> class set_type>
	class Einsum<bool, key_part_type, map_type, set_type> {
		using value_type = bool;

		using const_BoolHypertrie_t = const_BoolHypertrie<key_part_type, map_type, set_type>;
		using Join_t = Join<key_part_type, map_type, set_type>;
		using Operator_t = Operator<value_type, key_part_type, map_type, set_type>;
		using Entry_t = Entry<key_part_type, bool>;
		using Key_t = typename Entry_t::key_type;

		std::shared_ptr<Subscript> subscript{};
		std::shared_ptr<Context> context{};
		std::vector<const_BoolHypertrie_t> operands{};
		std::shared_ptr<Operator_t> op{};
		Entry_t entry{};

	public:
		Einsum(std::shared_ptr<Subscript> subscript, const std::vector<const_BoolHypertrie_t> &operands,
			   TimePoint timeout = std::numeric_limits<TimePoint>::max())
				: subscript(std::move(subscript)), context{std::make_shared<Context>(timeout)},
				  operands(operands),
				  op{Operator_t::construct(this->subscript, context)},
				  entry{false, Key_t(this->subscript->resultLabelCount(), std::numeric_limits<key_part_type>::max())} {
		    // set Properties von Einsum
		}

		[[nodiscard]] const std::shared_ptr<Subscript> &getSubscript() const {
			return subscript;
		}

		const std::vector<const_BoolHypertrie_t> &getOperands() const {
			return operands;
		}

		const std::shared_ptr<Operator_t> &getOp() const {
			return op;
		}

		struct iterator {
		private:


			std::shared_ptr<Operator_t> op;
			tsl::hopscotch_set<Key_t, ::einsum::internal::KeyHash<key_part_type>> found_entries{};
			Entry_t *current_entry;
			bool ended_ = false;
		public:
			iterator() = default;

			explicit iterator(Einsum &einsum, Entry_t &entry) : op(einsum.op), current_entry{&entry} {
				if (not op->ended()) {
					found_entries.insert(current_entry->key);
				}
			}

			iterator &operator++() {
				op->next();
				while (not op->ended()) {
					assert(current_entry->value == true);
					if (found_entries.find(current_entry->key) == found_entries.end()) {
						found_entries.insert(current_entry->key);
						return *this;
					}
					op->next();
				}
				return *this;
			}

			inline const Entry<key_part_type, value_type> &operator*() {
				return *current_entry;
			}

			inline const Entry<key_part_type, value_type> &value() {
				return *current_entry;
			}

			operator bool() const {
				return not op->ended();

			}

			[[nodiscard]] inline bool ended() const { return op->ended(); }

		};

		iterator begin() {
			op->load(operands, entry);
			return iterator{*this, entry};
		}

		[[nodiscard]] bool end() const {
			return false;
		}

		void clear() {
			throw std::logic_error("not yet implemented.");
		}
	};
}
#endif //HYPERTRIE_EINSUM_HPP
