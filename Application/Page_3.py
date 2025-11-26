import streamlit as st
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
from pymcdm.methods import TOPSIS, ARAS, EDAS, MABAC, PROMETHEE_II, CODAS, VIKOR, WASPAS, MARCOS, WSM
from pymcdm.correlations import weighted_spearman
from pymcdm.helpers import correlation_matrix
from pyDecision.algorithm import ahp_method
from pymcdm.weights import critic_weights
from pymcdm import visuals
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pastel_palette = [
            '#66c5cc',
            '#dcb0f2',
            '#f89c74',
            '#f6cf71',
            '#87c55f',
            '#e7d7ca',
            '#9eb9f3',
            '#fe88b1',
            '#c9db74',
            '#ddb398',
            '#e7d7ca'
        ]

st.title("Selecting the XAI method using MCDM")
st.divider()

df = None
uploaded_file = st.sidebar.file_uploader("UPLOAD YOUR DATA", type=["xlsx", "csv"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    try:
        if file_extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_extension == "csv":
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")


if df is not None:
    metrics_num = st.sidebar.number_input("ENTER A NUMBER OF METRICS", min_value=1, max_value=30, value=5)
    data = df.iloc[:, -metrics_num:]
    metric_name = data.columns.tolist()
    methods_name = df.iloc[:, 0]
    num_alternatives = len(data)
    weights = []

    st.write("## Loaded data")
    st.write(df)

    st.sidebar.write("ENTER TYPES OF CRITERIA")
    pills_results = []
    if metrics_num > 0:
        for i in metric_name:
            selected_pills = st.sidebar.pills(
                label=i,
                key=f"types_{i}",
                options=["Benefit", "Cost"],
                default="Benefit",
                help="Benefit: The higher, the better. Cost: The lower, the better."
            )

            if selected_pills== "Benefit":
                value = 1
            else:
                value = -1
            pills_results.append(value)

    weights_form = st.sidebar.radio(
        "METHOD FOR CALCULATING WEIGHTS",
        ["Direct rating", "Pairwise comparison (subjective)", "CRITIC (objective)"], index=0
    )

    if weights_form == "Direct rating":
        metric_values = {}
        for i in metric_name:
            value = st.sidebar.slider(
                i,
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                key=f"metric_slider_{i}",
                help=f"Enter a weight for the {i}."
            )

            metric_values[i] = value

        weights_list = list(metric_values.values())
        weights = np.array(weights_list)
        weights = normalize([weights], norm="l1")
        weights = weights.flatten()

        st.subheader("Normalized weights:")

        result = pd.DataFrame({
            'Metric': metric_name,
            'Weight': np.round(weights, 4)
        })
        st.write(result)


    elif weights_form == "Pairwise comparison (subjective)":
        st.sidebar.write("## Saaty's pairwise comparison")
        if metrics_num >= 10:
            st.sidebar.write("A higher number of metrics is not recommended for this method.")
        else:
            criteria = data.columns.tolist()

            comparison_matrix = np.ones((metrics_num, metrics_num))

            st.subheader("Calculated weights:")

            for i in range(metrics_num):
                for j in range(metrics_num):
                    if i == j:
                        comparison_matrix[i, j] = 1
                    elif i > j:  # Only show upper or lower triangle for input to avoid redundancy
                        comparison_matrix[i, j] = eval(
                            st.sidebar.selectbox(f"Compare {criteria[i]} to {criteria[j]}",
                                                 ("1", "3", "5", "7", "9", "1/3", "1/5", "1/7", "1/9")))
                        comparison_matrix[j, i] = 1 / comparison_matrix[i, j]

            weight_derivation = 'geometric'
            weights, rc = ahp_method(comparison_matrix, wd=weight_derivation)

            result = pd.DataFrame({
                'Metric': metric_name,
                'Weight': np.round(weights, 4)
            })
            st.write(result)

    else:  # weights_form == "CRITIC (objective)":
        criteria_types = np.ones(data.shape[1])

        weights = critic_weights(data, criteria_types)

        st.subheader("Calculated weights:")

        result = pd.DataFrame({
            'Metric': metric_name,
            'Weight': np.round(weights, 4)
        })

        st.write(result)

    if st.sidebar.button("Calculating preferences", icon=":material/calculate:", key="filter",
                         width="stretch", type="primary"):

        weights = np.array(weights)
        types = pills_results

        methods = [
            ARAS(),
            CODAS(),
            EDAS(),
            MABAC(),
            MARCOS(),
            PROMETHEE_II('usual'),
            TOPSIS(),
            VIKOR(),
            WASPAS(),
            WSM()
        ]

        method_names = ['ARAS', "CODAS", "EDAS", 'MABAC', "MARCOS","PROMETHEE_II",'TOPSIS', "VIKOR", "WASPAS",  "WSM"]

        prefs = []
        ranks = []
        df_pref = pd.DataFrame()
        df_rank = pd.DataFrame()
        df_rank["Method"] = methods_name
        first_col = df.columns[0]
        df_pref["Method"] = df[first_col]

        for method in methods:
            pref = method(data, weights, types)
            rank = method.rank(pref)

            prefs.append(pref)
            df_pref[f"Pref_{method.__class__.__name__}"] = pref
            ranks.append(rank)
            df_rank[f"{method.__class__.__name__}"] = rank

        all_rankings_modified = [
            [int(index - 1) for index in ranking]  # for Borda score
            for ranking in ranks
        ]
        df_rank.index = methods_name
        #df_rank.columns = method_names
        st.header('Results')
        st.subheader("Preferences")
        st.write(df_pref)
        st.subheader("Rankings")
        st.write(df_rank)

        def borda_count_aggregation(ranking_lists, candidate_names):

            num_alter = len(candidate_names)
            scores = {name: 0 for name in candidate_names}

            for single_ranking in ranking_lists:
                for rank_position, alternative_index in enumerate(single_ranking):
                    candidate_name = candidate_names[rank_position]
                    points = num_alter - 1 - alternative_index
                    scores[candidate_name] += points
            return scores

        final_borda_scores = borda_count_aggregation(all_rankings_modified, methods_name)
        borda_table = (
            pd.DataFrame(final_borda_scores.items(), columns=['Method', 'Score'])
            .sort_values(by='Score', ascending=False)
            .reset_index(drop=True)
        )

        st.write("### 🥇 Borda score")
        styled_df = borda_table.style.set_properties(
            subset = pd.IndexSlice[0, :],
            **{'background-color': '#b3a4dd', 'font-weight': 'bold', 'color': 'black'}
        )

        st.dataframe(styled_df)
        st.divider()

        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=pastel_palette)

        #---------------------------------------------------------------------------------------------------------------

        st.subheader("Visualisation of the rankings")

        df_long = df_rank.melt(
            id_vars='Method',
            var_name='MCDM Method',
            value_name='Position in ranking'
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.lineplot(
            data=df_long,
            x='MCDM Method',
            y='Position in ranking',
            hue='Method',
            palette=pastel_palette,
            marker='o',
            markersize=8,
            linewidth=2,
            ax=ax
        )

        plt.title('Position in Ranking Across MCDM Methods', fontsize=16)
        plt.xlabel('MCDM Method', fontsize=12)
        plt.ylabel('Position in ranking', fontsize=12)

        plt.gca().invert_yaxis()
        plt.yticks([1, 2, 3])

        plt.grid(True, linestyle='--', alpha=0.6)

        plt.legend(title='Methods')

        plt.tight_layout()
        st.pyplot(fig)

        #---------------------------------------------------------------------------------------------------------------

        st.divider()
        st.subheader("Visualization of the correlations between rankings")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            corr_matrix = correlation_matrix(np.array(ranks), weighted_spearman)
            plt.figure(figsize=(7, 7))
            visuals.correlation_heatmap(corr_matrix, labels=method_names, cmap="Greens")  #"PiYG" RdPu

            fig = plt.gcf()
            st.pyplot(fig)

        #---------------------------------------------------------------------------------------------------------------
        st.divider()


        def format_rank_to_string(rank_array, alternative_names) : #'A1 > A3 > A2'

            ranked_alt_names = []

            for r in range(1, len(rank_array) + 1):
                try:
                    alt_index = np.where(rank_array == r)[0][0]
                    ranked_alt_names.append(alternative_names[alt_index])
                except IndexError:
                    ranked_alt_names.append(f"ERROR_RANK{r}")

            return ' > '.join(ranked_alt_names)


        def sensitivity_analysis_all_w_change_multi(data, original_weights, _mcdm_methods_dict, types,
                                                    alternative_names):

            num_criteria = len(original_weights)
            all_results_sa = []

            delta_values = np.linspace(-0.40, 0.40, 20)

            for delta in delta_values:
                new_weights_unnormalized = np.abs(original_weights + delta)
                sum_of_new_w = np.sum(new_weights_unnormalized)

                final_weights = new_weights_unnormalized / sum_of_new_w if sum_of_new_w > 1e-6 else np.ones(
                    num_criteria) / num_criteria

                results = {'Weight Change (δ)': f'{delta:+.3f}'}

                for i in range(num_criteria):
                    results[f'C{i + 1}(w\'{i + 1})'] = f'{final_weights[i]:.4f}'

                for method_name, method_instance in mcdm_methods_dict.items():
                    try:
                        pref = method_instance(data, final_weights, types)
                    except TypeError:
                        pref = method_instance(data, final_weights)

                    rank = method_instance.rank(pref)
                    rank_string = format_rank_to_string(rank, alternative_names)
                    results[method_name] = rank_string
                    results[f"{method_name}_rank"] = str(rank.tolist())

                all_results_sa.append(results)

            return pd.DataFrame(all_results_sa)


        alternative_names = [f'A{i + 1}' for i in range(num_alternatives)]
        mcdm_methods_dict = {
            f"{name} Rank": method
            for name, method in zip(method_names, methods)
        }

        df_sensitivity = sensitivity_analysis_all_w_change_multi(
            data,
            weights,
            mcdm_methods_dict,
            types,
            alternative_names
        )


        if not df_sensitivity.empty:

            st.header("Sensitivity Analysis")
            st.markdown(f"Analyzed Alternatives: **{', '.join(alternative_names)}**")

            column_order = ['Weight Change (δ)']
            for i in range(metrics_num):
                column_order.append(f'C{i + 1}(w\'{i + 1})')
            column_order.extend(mcdm_methods_dict.keys())

            df_sensitivity = df_sensitivity[column_order]

            st.dataframe(df_sensitivity, height=35+20*35)

        #---------------------------------------------------------------------------------------------------------------

        def evaluate_first_rank_percentage(df_sensitivity, alternative_names):
            """
            It analyzes the DF and calculates the percentage of times each alternative held 1st place.
            """
            rank_columns = [col for col in df_sensitivity.columns if ' Rank' in col]

            if not rank_columns:
                return pd.DataFrame()

            df_evaluation = pd.DataFrame(index=alternative_names)
            total_experiments = len(df_sensitivity)

            for rank_col in rank_columns:

                df_sensitivity['Winner'] = df_sensitivity[rank_col].apply(lambda x: x.split(' > ')[0])
                winner_counts = df_sensitivity['Winner'].value_counts()
                winner_percentages = (winner_counts / total_experiments) * 100

                df_evaluation[rank_col] = winner_percentages

                df_evaluation = df_evaluation.fillna(0)
                df_evaluation[rank_col] = df_evaluation[rank_col].apply(lambda x: f"{x:.2f} %")

            return df_evaluation

        df_robustness = evaluate_first_rank_percentage(df_sensitivity, alternative_names)

        st.subheader("Percentage of 1st Rank Across Experiments")
        st.dataframe(df_robustness, use_container_width=True)

        #---------------------------------------------------------------------------------------------------------------

        rank_columns = list(mcdm_methods_dict.keys())

        df_all_ranks_long = []

        for index, row in df_sensitivity.iterrows():

            # For each method (column)
            for method_col in rank_columns:

                rank_string = row[method_col]
                # Assuming rank_string format is 'A1 > A3 > A2'
                ordered_alts = rank_string.split(' > ')

                # For each alternative
                for i, alt_name_temp in enumerate(alternative_names):  # alt_name_temp is 'A1', 'A2', etc.

                    # Finding the Rank (1, 2, 3...) of the alternative
                    try:
                        # Get the position (0-indexed) and convert to rank (1-indexed)
                        rank_value = ordered_alts.index(alt_name_temp) + 1
                    except ValueError:
                        # Alternative not found in the ranking string
                        continue

                    df_all_ranks_long.append({
                        'Alternative': alt_name_temp,
                        'Rank': rank_value,
                        'MCDM Method': method_col.replace(' Rank', ''),
                        'Delta': row['Weight Change (δ)']
                    })

        df_ranks_distribution = pd.DataFrame(df_all_ranks_long)

        st.subheader("Pie Chart: Overall Share of 1st Rank Wins")

        actual_alternative_names = methods_name.tolist()

        if len(alternative_names) == len(actual_alternative_names):
            # Map temporary names (A1, A2, ...) to actual names (IG, SG, ...)
            legend_map = {
                temp_name: actual_name
                for temp_name, actual_name in zip(alternative_names, actual_alternative_names)
            }
        else:
            st.error("Error: Mismatch between number of temporary names (A1, A2...) and actual alternative names.")
            legend_map = {}  # Use an empty map to prevent replacement

        # Step 2: Filtering data: Count only rows where Rank = 1
        df_winners = df_ranks_distribution[
            df_ranks_distribution['Rank'] == 1].copy()

        # Step 3: RENAMING ALTERNATIVES in the data
        # Replace A1, A2, A3, A4 with the actual names using the dynamically created map
        if legend_map:
            df_winners['Alternative'] = df_winners['Alternative'].replace(legend_map)

        # Step 4: Counting total wins for each alternative
        # Counting is done using the renamed names
        total_wins = df_winners['Alternative'].value_counts()

        if not total_wins.empty:

            # Step 5: Plotting the Pie Chart
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))

            # Automatic percentage formatting (autopct='%1.1f%%')
            # Labels automatically use the renamed names (IG, SG, etc.)
            ax_pie.pie(
                total_wins.values,
                labels=total_wins.index,  # Index contains the renamed names
                autopct='%1.1f%%',
                startangle=90,
            )

            ax_pie.axis('equal')  # Ensures the pie chart is circular
            ax_pie.set_title('Overall Share of 1st Rank Wins Across All Methods and $\delta$ Values', fontsize=10)

            # Display the plot in the center column
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig_pie)
        else:
            st.warning("No alternative achieved 1st rank.")
            
        #---------------------------------------------------------------------------------------------------------------

        def plot_rank_change_sensitivity(df_sensitivity, target_method_col, alternative_names, num_alternatives):
            """
            Creates and displays a plot of alternative rank change depending on the delta parameter (weight change).

            :param df_sensitivity: DataFrame containing sensitivity analysis results.
            :param target_method_col: The column name in df_sensitivity containing the ranking
                              (e.g., 'TOPSIS Rank', format: 'A1 > A3 > A2').
            :param alternative_names: List of alternative names.
            :param num_alternatives: Total number of alternatives.
            """

            if target_method_col not in df_sensitivity.columns:
                st.error(f"Error: Ranking column '{target_method_col}' was not found.")
                return

            df_rank_change = df_sensitivity[['Weight Change (δ)', target_method_col]].copy()

            df_rank_change['Weight Change (δ)'] = (
                df_rank_change['Weight Change (δ)']
                .astype(str)
                .str.replace('+', '', regex=False)
                .astype(float)
            )
            df_rank_change = df_rank_change.sort_values('Weight Change (δ)').reset_index(drop=True)


            def get_rank_for_alt(rank_string, target_alt):
                # rank_string is 'A1 > A3 > A2'
                ordered_alts = rank_string.split(' > ')
                try:
                    return ordered_alts.index(target_alt) + 1
                except ValueError:
                    return np.nan

            for alt in alternative_names:
                df_rank_change[f'Rank_{alt}'] = df_rank_change[target_method_col].apply(
                    lambda x: get_rank_for_alt(x, alt)
                )

            df_long_ranks = df_rank_change.melt(
                id_vars='Weight Change (δ)',
                value_vars=[f'Rank_{alt}' for alt in alternative_names],
                var_name='Alternatíva',
                value_name='Rank'
            ).dropna(subset=['Rank'])

            df_long_ranks['Alternatíva'] = df_long_ranks['Alternatíva'].str.replace('Rank_', '')

            fig, ax = plt.subplots(figsize=(12, 6))
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=pastel_palette)

            sns.lineplot(
                data=df_long_ranks,
                x='Weight Change (δ)',
                y='Rank',
                hue='Alternatíva',
                marker='o',
                markersize=6,
                linewidth=2,
                ax=ax
            )

            method_name_clean = target_method_col.replace(' Rank', '')
            ax.set_title(f'Rank Change of Alternatives due to ($\delta$) ({method_name_clean})', fontsize=16)
            ax.set_ylabel('Rank', fontsize=12)
            ax.set_xlabel('Additive Weight Change ($\delta$)', fontsize=12)

            # Invertovanie Y-osi tak, aby Rank 1 bol hore
            ax.set_yticks(np.arange(1, num_alternatives + 1))
            ax.invert_yaxis()

            ax.grid(axis='both', linestyle='--', alpha=0.5)
            ax.legend(title='Methods', loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout()

            st.pyplot(fig)

        with st.expander("Rank change of alternatives due to δ for method ARAS"):
            plot_rank_change_sensitivity(
                df_sensitivity,
                "ARAS Rank",
                alternative_names,
                num_alternatives
            )

        with st.expander("Rank change of alternatives due to δ for method CODAS"):
            st.write("aaaaaaaaaaaaaaaaaaa")
            plot_rank_change_sensitivity(
                df_sensitivity,
                "CODAS Rank",
                alternative_names,
                num_alternatives
            )

        with st.expander("Rank change of alternatives due to δ for method EDAS"):
            plot_rank_change_sensitivity(
                df_sensitivity,
                "EDAS Rank",
                alternative_names,
                num_alternatives
            )

        with st.expander("Rank change of alternatives due to δ for method MABAC"):
            plot_rank_change_sensitivity(
                df_sensitivity,
                "MABAC Rank",
                alternative_names,
                num_alternatives
            )

        with st.expander("Rank change of alternatives due to δ for method MARCOS"):
            plot_rank_change_sensitivity(
                df_sensitivity,
                "MARCOS Rank",
                alternative_names,
                num_alternatives
            )

        with st.expander("Rank change of alternatives due to δ for method PROMETHEE II"):
            plot_rank_change_sensitivity(
                df_sensitivity,
                "PROMETHEE_II Rank",
                alternative_names,
                num_alternatives
            )

        with st.expander("Rank change of alternatives due to δ for method TOPSIS"):
            plot_rank_change_sensitivity(
                df_sensitivity,
                "TOPSIS Rank",
                alternative_names,
                num_alternatives
            )

        with st.expander("Rank change of alternatives due to δ for method VIKOR"):
            plot_rank_change_sensitivity(
                df_sensitivity,
                "VIKOR Rank",
                alternative_names,
                num_alternatives
            )

        with st.expander("Rank change of alternatives due to δ for method WASPAS"):
            plot_rank_change_sensitivity(
                df_sensitivity,
                "WASPAS Rank",
                alternative_names,
                num_alternatives
            )

        with st.expander("Rank change of alternatives due to δ for method WSM"):
            plot_rank_change_sensitivity(
                df_sensitivity,
                "WSM Rank",
                alternative_names,
                num_alternatives
            )

