import os


def _retrieve_dannce_mat(root):
    candidates = sorted([p for p in os.listdir(root) if p.endswith(".mat")])
    datapaths = [os.path.join(root, dp) for dp in candidates]
    datapaths = [dp for dp in datapaths if os.path.exists(dp)]
    return datapaths, candidates


def _retrieve_by_template(root, template):
    candidates = sorted([p for p in os.listdir(root) if p[0].isdigit()])
    datapaths = [
        os.path.join(root, dp, template)
        for dp in candidates
    ]
    datapaths = [dp for dp in datapaths if os.path.exists(dp)]
    return datapaths, candidates


def _retrieve_dannce_basic(root):
    return _retrieve_by_template(root, "DANNCE/predict/save_data_AVG0.mat")


def _retrieve_dannce_lone(root):
    return _retrieve_by_template(root, "SDANNCE/bsl0.5_FM/save_data_AVG0.mat")


def _retrieve_dannce_social(root):
    candidates1, datapaths1 = _retrieve_by_template(root, "SDANNCE/bsl0.5_FM_rat1/save_data_AVG0.mat")
    candidates2, datapaths2 = _retrieve_by_template(root, "SDANNCE/bsl0.5_FM_rat2/save_data_AVG0.mat")
    candidates = sorted(list((set(candidates1) & set(candidates2))))
    datapaths = datapaths1 + datapaths2
    return datapaths, candidates


def _retrieve_moseq(root):
    datapaths = candidates = [root]
    return datapaths, candidates