def borda_score(rankings, common_labels):
    """
    Calculates the Borda score for classes based on rankings, handling ties.

    :param rankings: A list of rankings of classes across different instances.
                     Each element of the list is a dictionary with class names as keys and unit counts as values.
                     Example: [{'A': 3, 'B': 2, 'C': 0, 'D': 1}, ...]
    :param common_labels: A list of classes common between the models.
    :return: A dictionary containing the total Borda score for each class.
    """
    total_scores = {}

    for ranking in rankings:
        # Sort the classes by unit count (highest to lowest)
        sorted_classes = sorted(
            ranking.items(), key=lambda x: x[1], reverse=True)

        # Initialize scores for each rank
        n = len(common_labels)  # Number of common classes
        scores = [n - i - 1 for i in range(n)]  # Borda scores (n-1, n-2,..., 0)

        # Handle ties
        current_rank = 0
        while current_rank < len(sorted_classes):
            same_rank = [sorted_classes[current_rank]]
            for next_rank in range(current_rank + 1, len(sorted_classes)):
                if sorted_classes[next_rank][1] == sorted_classes[current_rank][1]:
                    same_rank.append(sorted_classes[next_rank])
                else:
                    break

            # Calculate the average score for tied classes
            avg_score = sum(
                scores[current_rank:current_rank + len(same_rank)]) / len(same_rank)

            # Assign the score to tied classes
            for cls, _ in same_rank:
                if cls in total_scores:
                    total_scores[cls] += avg_score
                else:
                    total_scores[cls] = avg_score

            current_rank += len(same_rank)

    return total_scores