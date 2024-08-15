from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Function to get fuzzy numbers based on linguistic ratings
def get_fuzzy_number(rating):
    fuzzy_scale = {
        'VP': (0, 0, 0.25),
        'P': (0, 0.25, 0.5),
        'F': (0.25, 0.5, 0.75),
        'G': (0.5, 0.75, 1),
        'VG': (0.75, 1, 1)
    }
    return fuzzy_scale.get(rating, (0, 0, 0))

# Calculate fuzzy weights for each OA
def calculate_oa_weights(ratings):
    oa_weights = {}
    for oa, rate_list in ratings.items():
        a_values = [get_fuzzy_number(rate)[0] for rate in rate_list]
        b_values = [get_fuzzy_number(rate)[1] for rate in rate_list]
        c_values = [get_fuzzy_number(rate)[2] for rate in rate_list]
        a = min(a_values)
        b = round(float(np.mean(b_values)), 4)
        c = max(c_values)
        oa_weights[oa] = (round(a, 4), b, round(c, 4))
    return oa_weights

# Calculate the fuzzy decision matrix for each FA under each OA by each evaluator
def calculate_fuzzy_decision_matrix(fa_ratings):
    fuzzy_decision_matrix = {}
    for fa, oa_dict in fa_ratings.items():
        fuzzy_decision_matrix[fa] = {}
        for oa, evaluator_ratings in oa_dict.items():
            a_values = [get_fuzzy_number(rate)[0] for rate in evaluator_ratings]
            b_values = [get_fuzzy_number(rate)[1] for rate in evaluator_ratings]
            c_values = [get_fuzzy_number(rate)[2] for rate in evaluator_ratings]
            a = min(a_values)
            b = round(float(np.mean(b_values)), 4)
            c = max(c_values)
            fuzzy_decision_matrix[fa][oa] = (round(a, 4), b, round(c, 4))
    return fuzzy_decision_matrix

# Calculate the weighted fuzzy decision matrix
def calculate_weighted_fuzzy_decision_matrix(fuzzy_decision_matrix, oa_weights):
    weighted_fuzzy_decision_matrix = {}
    for fa, oa_dict in fuzzy_decision_matrix.items():
        weighted_fuzzy_decision_matrix[fa] = {}
        for oa, values in oa_dict.items():
            weight = oa_weights[oa]
            weighted_a = round(weight[0] * values[0], 4)
            weighted_b = round(weight[1] * values[1], 4)
            weighted_c = round(weight[2] * values[2], 4)
            weighted_fuzzy_decision_matrix[fa][oa] = (weighted_a, weighted_b, weighted_c)
    return weighted_fuzzy_decision_matrix

# Calculate FPIS and FNIS
def calculate_fpis_fnis(weighted_fuzzy_decision_matrix):
    fpis = {}
    fnis = {}
    
    for fa, oa_dict in weighted_fuzzy_decision_matrix.items():
        a_values = [a for (a, b, c) in oa_dict.values()]
        c_values = [c for (a, b, c) in oa_dict.values()]

        fpis[fa] = (max(c_values), max(c_values), max(c_values))
        fnis[fa] = (min(a_values), min(a_values), min(a_values))

    return fpis, fnis

# Calculate distance between two fuzzy numbers
def fuzzy_distance(v1, v2):
    return np.sqrt((1/3)*((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 + (v1[2] - v2[2])**2))

# Calculate distances from FPIS and FNIS, and similarity coefficient (SC)
def calculate_distances_and_sc(weighted_fuzzy_decision_matrix, fpis, fnis):
    distances_from_fpis = {}
    distances_from_fnis = {}
    similarity_coefficients = {}

    for fa, oa_dict in weighted_fuzzy_decision_matrix.items():
        d_i_star = sum(fuzzy_distance(weighted_fuzzy_decision_matrix[fa][oa], fpis[fa]) for oa in oa_dict)
        d_i_minus = sum(fuzzy_distance(weighted_fuzzy_decision_matrix[fa][oa], fnis[fa]) for oa in oa_dict)

        distances_from_fpis[fa] = round(d_i_star, 4)
        distances_from_fnis[fa] = round(d_i_minus, 4)

        sc_i = round(d_i_minus / (d_i_star + d_i_minus), 4) if (d_i_star + d_i_minus) != 0 else 0
        similarity_coefficients[fa] = sc_i

    return distances_from_fpis, distances_from_fnis, similarity_coefficients

# Rank the alternatives based on similarity coefficients
def rank_alternatives(similarity_coefficients):
    sorted_alternatives = sorted(similarity_coefficients.items(), key=lambda item: item[1], reverse=True)
    rankings = {alternative: rank+1 for rank, (alternative, _) in enumerate(sorted_alternatives)}
    return rankings

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_evaluators = int(request.form['num_evaluators'])
        num_oas = int(request.form['num_oas'])
        num_fas = int(request.form['num_fas'])

        ratings = {}
        for i in range(num_oas):
            oa_name = f'OA{i+1}'
            ratings[oa_name] = [request.form[f'{oa_name}_eval{j+1}'] for j in range(num_evaluators)]

        fa_ratings = {}
        for i in range(num_fas):
            fa_name = f'FA{i+1}'
            fa_ratings[fa_name] = {}
            for j in range(num_oas):
                oa_name = f'OA{j+1}'
                fa_ratings[fa_name][oa_name] = [request.form[f'{fa_name}_{oa_name}_eval{k+1}'] for k in range(num_evaluators)]

        # Calculate fuzzy weights for each OA
        oa_weights = calculate_oa_weights(ratings)

        # Calculate the fuzzy decision matrix for each FA under each OA by each evaluator
        fuzzy_decision_matrix = calculate_fuzzy_decision_matrix(fa_ratings)

        # Calculate the weighted fuzzy decision matrix
        weighted_fuzzy_decision_matrix = calculate_weighted_fuzzy_decision_matrix(fuzzy_decision_matrix, oa_weights)

        # Calculate FPIS and FNIS
        fpis, fnis = calculate_fpis_fnis(weighted_fuzzy_decision_matrix)

        # Calculate distances and similarity coefficient (SC)
        distances_from_fpis, distances_from_fnis, similarity_coefficients = calculate_distances_and_sc(weighted_fuzzy_decision_matrix, fpis, fnis)

        # Rank alternatives
        rankings = rank_alternatives(similarity_coefficients)

        return render_template('results.html', oa_weights=oa_weights, fuzzy_decision_matrix=fuzzy_decision_matrix, 
                               weighted_fuzzy_decision_matrix=weighted_fuzzy_decision_matrix, fpis=fpis, fnis=fnis, 
                               distances_from_fpis=distances_from_fpis, distances_from_fnis=distances_from_fnis, 
                               similarity_coefficients=similarity_coefficients, rankings=rankings)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)