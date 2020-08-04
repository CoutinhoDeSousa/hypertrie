#ifndef HYPERTRIE_CARDINALITYESTIMATION_HPP
#define HYPERTRIE_CARDINALITYESTIMATION_HPP

#include <cmath>
#include "Dice/einsum/internal/Subscript.hpp"
#include "Dice/einsum/internal/Entry.hpp"
#include "Context.hpp"
extern "C"
{

#include <igraph/igraph.h>
#include <igraph/igraph_cliques.h>  // import for 3.3 igraph_maximal_independent_vertex_sets
}


namespace einsum::internal {
// enum local for threestate
    enum SORT{
        MINIMUM,
        MAXIMUM,
        RANDOM
    };
    enum WEIGHT{
        NORMAL,
        CARDINALITY
    };

	template<typename key_part_type, template<typename, typename> class map_type,
			template<typename> class set_type>
	struct CardinalityEstimation {
		using const_BoolHypertrie_t = const_BoolHypertrie<key_part_type, map_type, set_type>;

		/**
		 * Method to get the Label with the smallest cardinality (first found if many have the same cardinality)
		 * @param operands
		 * @param label_candidates
		 * @param sc
		 * @return
		 */
		static Label getMinCardLabel(const std::vector<const_BoolHypertrie_t> &operands,
		                             const std::shared_ptr<Subscript> &sc,
		                             std::shared_ptr<Context> context, SORT sort = MINIMUM, WEIGHT weight = CARDINALITY) {
            // TODO: Min , Max; random als eine Variante
            // Todo: calcCard oder weight als gewicht
            const tsl::hopscotch_set <Label> &operandsLabelSet = sc->getOperandsLabelSet();
            Label returnLabel;
            auto bestLabel = context->best_label.find(*sc);
            if (bestLabel == context->best_label.end()) // There is no best Label
            {

                // get Candidate Set
                std::set<Label> candidates;
                //                if (context->label_candidates.find(*sc) == context->label_candidates.end()) {
                if (!context->label_candidates.count(*sc)) {
                    // Weight function berechnen und steuerungsvariablen setzen
                    //test if we need this
                    std::map<char,int> edgesList;
                    std::map<int,int> weights;
                    getWeights(operands,edgesList,weights, weight, sc);
                    // getMwis aufrufen
                    candidates = getMWIS(   sc->getRawSubscript().operands, weights, edgesList); // TODO: hier: map mit weights mitgeben falls nicht die standard weight funktion verwendet wird
                    // TODO: anschließend sortieren
                    // candidates sortieren anhand sort mit weightmap
                    // Controllvariable to save it in labelcandidates or not
                    context->label_candidates[*sc] = candidates;
                } else {
                    candidates = context->label_candidates[*sc];
                }

                // Take best Label (atm the first)
                // if its filled at all
                auto best_candidate = candidates.begin();
                if(candidates.size() > 0){
                context->best_label[*sc] = *best_candidate;
                }
                if (candidates.size() > 1) { // TODO: only update if not there already
                    // deletes the first label for the sub-subscript
                    auto start = std::next(candidates.begin(), 1);
                    context->label_candidates[*sc->removeLabel(*best_candidate)] =
                            std::set<Label>{start, candidates.end()};
                } else if (sc->getRawSubscript().operands.size() == 0) {
                    // fill with empty label candidates
                    context->label_candidates[*sc->removeLabel(*best_candidate)] =
                            std::set<Label>{};
                }

            }
            // only if its filled
            if(context->best_label.find(*sc) != context->best_label.end()){
               return context->best_label[*sc];
            }
            // BaseCase as before
            if (operandsLabelSet.size() == 1){
                return *operandsLabelSet.begin();
            }
            throw std::logic_error{"should never be reached."};
             return returnLabel; // für den fall das zuviel weggenommen wurde
            //return min_label;


		}


	protected:
		/**
		 * Calculates the cardinality of an Label in a step.
		 * @tparam T type of the values hold by processed Tensors (Tensor).
		 * @param operands Operands for this Step.
		 * @param step current step
		 * @param label the label
		 * @return label's cardinality in current step.
		 */
		static double calcCard(const std::vector<const_BoolHypertrie_t> &operands, const Label label,
		                       const std::shared_ptr<Subscript> &sc, bool min = true) {
			// get operands that have the label
			const std::vector<LabelPos> &op_poss = sc->getPossOfOperandsWithLabel(label);
			std::vector<double> op_dim_cardinalities(op_poss.size(), 1.0);
			auto label_count = 0;
			auto best_dim_card = (min) ? std::numeric_limits<size_t>::max() : std::numeric_limits<size_t>::min();
			tsl::hopscotch_set <size_t> sizes{};

			const LabelPossInOperands &label_poss_in_operands = sc->getLabelPossInOperands(label);
			// iterate the operands that hold the label
			for (auto[i, op_pos] : iter::enumerate(op_poss)) {
				const auto &operand = operands[op_pos];
				const auto op_dim_cards = operand.getCards(label_poss_in_operands[op_pos]);
				const auto best_op_dim_card = (min) ? *std::min_element(op_dim_cards.cbegin(), op_dim_cards.cend())
				        : *std::max_element(op_dim_cards.cbegin(), op_dim_cards.cend());


				for (const auto &op_dim_card : op_dim_cards)
					sizes.insert(op_dim_card);

				label_count += op_dim_cards.size();
                // update minimal dimenension cardinality#

				if ((min and best_op_dim_card < best_dim_card) or (not min and best_op_dim_card > best_dim_card))
                    best_dim_card = best_op_dim_card;

				op_dim_cardinalities[i] = double(best_op_dim_card); //
			}

			std::size_t max_op_size = 0;
			std::vector<std::size_t> op_sizes{};
			for (auto op_pos : op_poss) {
				auto current_op_size = operands[op_pos].size();
				op_sizes.push_back(current_op_size);
				if (current_op_size > max_op_size)
					max_op_size = current_op_size;
			}

			auto dim_card_d = double(best_dim_card);

			double card = std::accumulate(op_dim_cardinalities.cbegin(), op_dim_cardinalities.cend(), double(1),
			                              [&](double a, double b) {
				                              return a * dim_card_d / b;
			                              }) / sizes.size();
			return card;

		}

		static void getWeights(const std::vector<const_BoolHypertrie_t> &operands,std::map<char,int> &edgesList,
        std::map<int,int> &weights, WEIGHT weight,const std::shared_ptr<Subscript> &sc){

            int id=0;
                for (const auto &operand_sc : sc->getRawSubscript().operands) {
                    for (const auto &op_for_weight: operand_sc) {
                        std::pair<std::map<char, int>::iterator, bool> itNodes;
                        itNodes = edgesList.insert(std::pair<char, int>(op_for_weight, id));

                        if (itNodes.second) { //added new
                            weights.insert(std::pair<int, int>(id, 1));
                            id++;
                        } else { //already in
                            int foundIndex = std::distance(edgesList.begin(), itNodes.first);
                            weights[foundIndex]++;
                        }
                    }
                }
            if(weight == CARDINALITY){
                const tsl::hopscotch_set <Label> &operandsLabelSet = sc->getOperandsLabelSet();
                const tsl::hopscotch_set <Label> &lonely_non_result_labels = sc->getLonelyNonResultLabelSet();
                for (const Label label : operandsLabelSet) {
                    //auto found = edgesList.find()
                    if (lonely_non_result_labels.count(label)){
                        // lonely weight auf 0 setzen?
                        weights[findIDofLabel(weights, edgesList, label)] = 0;
                        continue;
                    }
                    // find weight
                    weights[findIDofLabel(weights,edgesList,label)] = calcCard(operands, label, sc);

                }

            }
		}
		/**
		 * returns id of Label in the given maps;
		 * @param weights
		 * @param edgesList
		 * @param label
		 * @return
		 */
		static auto findIDofLabel(std::map<int,int> &weights, std::map<char,int> &edgesList, Label label){
		    auto iterator = edgesList.find(label);
		    if( iterator != edgesList.end()){
		        return weights.find(iterator->second)->second;

		    }

            throw std::logic_error{"ID of Label couldn't be found"};
		}


        /**
         * returns for given Operands List the (first) maximum weighted independent set
         * @param operands operand list
         * @return Labelset with the mwis
         */
	    static std::set<Label> getMWIS(const einsum::internal::OperandsSc &operands, std::map<int,int> &weights, std::map<char,int> &edgesList){
// Todo: Weights als argument hinzufügen
            igraph_t graph;
            igraph_vector_t v;
            igraph_vector_ptr_t mis = IGRAPH_VECTOR_PTR_NULL;
            igraph_vector_ptr_init(&mis, 0);
            /*
             * Start building a graph
             */
            std::vector<int> edges;
            /*
            std::map<char,int> edgesList;
            std::map<int,int> weights;
            // zähle 2 maps hoch (label zu id & id zu weight)
            int id = 0;
           */
            for (const auto &operand_sc : operands) {
                if(operand_sc.size() == 2){ // just 1 edge, 2 nodes
                    //todo: fehler kontrolle wenn nicht gefunden (füge sie ja oben ein, deshalb nochmal anspechen)
                    //todo: in Methode auslagern
                    edges.push_back(edgesList[operand_sc[0]]);
                    edges.push_back(edgesList[operand_sc[1]]);
                }
                if(operand_sc.size() == 3){ // 3 edges
                    edges.push_back(edgesList[operand_sc[0]]);
                    edges.push_back(edgesList[operand_sc[1]]);
                    edges.push_back(edgesList[operand_sc[0]]);
                    edges.push_back(edgesList[operand_sc[2]]);
                    edges.push_back(edgesList[operand_sc[1]]);
                    edges.push_back(edgesList[operand_sc[2]]);

                }


            }


            int sizeArray = edges.size();
            igraph_real_t e[sizeArray];
            std::copy(edges.begin(),edges.end(),e);
            igraph_vector_view(&v, e, sizeArray);
            igraph_create(&graph,&v, 0 , IGRAPH_UNDIRECTED);

            igraph_maximal_independent_vertex_sets(&graph,&mis);


            // find the "best" mis
            size_t best_mis_id = 0;
            size_t best_size = 0;

            // bestes auswählen anhat der weights (die ich noch nciht habe, aufzaehlen in der for oben
            const size_t set_number_count = igraph_vector_ptr_size(&mis);
            for (size_t i = 0 ; i < set_number_count; i++){
                igraph_vector_t *tmp_mis = (igraph_vector_t*) igraph_vector_ptr_e(&mis,long(i));

                size_t tmp_mis_size = igraph_vector_size(tmp_mis); //const??
                size_t x = 0;
                for (size_t j = 0 ; j < tmp_mis_size; j++){
                    x+= weights[size_t(igraph_vector_e(tmp_mis, long(j)))];
                    if(best_size < x) {
                        best_size = x;
                        best_mis_id = i;
                    }
                }

            }


            igraph_vector_t *best_mis = (igraph_vector_t*)igraph_vector_ptr_e(&mis, long(best_mis_id)); // 0 ersetzen durch "bestes"
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
                igraph_vector_ptr_destroy(&mis); //trz aufräumen und dann weiterschmeissen
                throw error;
            }

            igraph_destroy(&graph);
            igraph_vector_ptr_destroy(&mis);

            /*
             * End building a graph
             */
            // retransformation to labels

            std::vector<char> vmis_labels;
            std::map<char,int>::const_iterator reverseIt;
            for (auto &[labelx, node_idx] : edgesList){
                auto found = std::find(vmis.begin(), vmis.end(), node_idx);
                if (found != vmis.end())
                    vmis_labels.push_back(labelx);
            }

            return std::set<Label>{vmis_labels.begin(),vmis_labels.end()};
        }
	};
}
#endif //HYPERTRIE_CARDINALITYESTIMATION_HPP
