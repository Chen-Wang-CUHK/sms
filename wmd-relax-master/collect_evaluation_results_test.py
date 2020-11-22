import os
import csv
import argparse


def collect_eval_results(opts):
    log_dir = opts.log_folder
    saved_dir = opts.saved_folder
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    log_files = []
    if os.path.isdir(log_dir):
        for sub_dir in ['tac_08', 'tac_09', 'tac_2010', 'tac_2011', 'tac_cnndm']:
            sub_log_dir = os.path.join(log_dir, sub_dir)
            log_files = log_files + [f for f in os.listdir(sub_log_dir)]
    elif os.path.isfile(log_dir):
        log_files = [log_dir]
    else:
        print("The log_dir is invalid: {}".format(log_dir))
        raise NotImplementedError

    log_files = sorted(log_files)

    saved_csv = os.path.join(saved_dir, "{}_collected.csv".format(opts.csv_name_base))
    assert not os.path.isfile(saved_csv), "The {} already exists! Make sure your saved_folder is correctly configured.".format(saved_csv)
    print("Evaluation results are saved in {}".format(saved_csv))

    fieldnames = ['model_name',
                  'tac_2011_pearson', 'tac_2011_spearman', 'tac_2011_kendall', 'blank1',
                  'tac_2010_pearson', 'tac_2010_spearman', 'tac_2010_kendall', 'blank2',
                  'tac_09_pearson', 'tac_09_spearman', 'tac_09_kendall', 'blank3',
                  'tac_08_pearson', 'tac_08_spearman', 'tac_08_kendall', 'blank3',
                  'tac_cnndm_overall_pearson', 'tac_cnndm_overall_spearman', 'tac_cnndm_overall_kendall', 'blank5',
                  'tac_cnndm_grammar_pearson', 'tac_cnndm_grammar_spearman', 'tac_cnndm_grammar_kendall', 'blank6',
                  'tac_cnndm_redundancy_pearson', 'tac_cnndm_redundancy_spearman', 'tac_cnndm_redundancy_kendall'
                  ]

    with open(saved_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        rslt_dict = {}
        required_hss_type = {'pyr_score', 'overall', 'grammar', 'redundancy'}
        ave_type = {'pyr_score': 'macro', 'overall': 'micro', 'grammar': 'micro', 'redundancy': 'micro'}
        required_corr_type = {'pearson', 'spearman', 'kendall'}
        for f in log_files:
            # sub_dir = f.split('.')[0] + '_' + f.split('.')[1]
            sub_dir = f.split('_')[0] + '_' + f.split('_')[1]
            log_file = open(os.path.join(log_dir, sub_dir, f))
            print("Preprocessing {}".format(os.path.join(log_dir, f)))

            model_name = 'mvrsc.' + f[len(sub_dir)+1:-len('_log.txt')]
            if model_name not in rslt_dict:
                rslt_dict[model_name] = {}
                for k in fieldnames:
                    rslt_dict[model_name][k] = ''
                rslt_dict[model_name]['model_name'] = model_name

            for line in log_file:
                line = line.strip()
                for hss_type in required_hss_type:
                    if hss_type in line:
                        for corr_metric in required_corr_type:
                            corr_name = hss_type + '_' + ave_type[hss_type] + '_' + corr_metric
                            if corr_name in line:
                                mean_r = float(line.strip().split(':')[-1].strip()[:-1])
                                mean_r = round(mean_r, 4)
                                if hss_type == 'pyr_score':
                                    rslt_dict[model_name][sub_dir + '_' + corr_metric] = mean_r
                                else:
                                    rslt_dict[model_name][sub_dir + '_' + hss_type + '_' + corr_metric] = mean_r
                                break
        for model_name in rslt_dict:
            writer.writerow(rslt_dict[model_name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="collect_evaluation_results.py")
    parser.add_argument("--log_folder", "-log_folder", type=str, default='logs',
                        help="The folder that stored the evaluation logs.")
    parser.add_argument("--saved_folder", "-saved_folder", type=str, default='logs',
                        help="The folder to save the collected evaluation results.")
    parser.add_argument("--csv_name_base", "-csv_name_base", type=str, default='smd',
                        help="The name base of the saved .csv file.")
    opts = parser.parse_args()

    collect_eval_results(opts)