import csv
import sys
import numpy as np
import regex as re
import matplotlib as mpl
mpl.use('Agg')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']


import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import itertools
import argparse
import pickle
import bench
import numpy.random as random 


# Import nimfa library entry point for factorization
import nimfa

import logging

import protein_complex_maps.normalization_util as nu
import protein_complex_maps.protein_util as pu
import protein_complex_maps.bicluster.bicluster as bc
import protein_complex_maps.plots.plot_bicluster as pbc
import protein_complex_maps.random_sampling_util as rsu
import protein_complex_maps.score_util as su
import protein_complex_maps.bicluster_generator as bg
import protein_complex_maps.annealer as anl

logging.basicConfig(level = logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')


def load_complex_portal(path):
    UP = re.compile(r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$')
    out = {}
    with open(path) as f:
        rows = list(csv.reader(f, delimiter='\t'))
        hdr = rows[0]
        iexp = hdr.index("Expanded participant list")
        iac = hdr.index("#Complex ac")
        for r in rows[1:]:
            groups = []
            for tok in r[iexp].split("|"):
                acc= tok.split('(')[0]
                if acc.startswith('[') and acc.endswith(']'):
                    mem = [m for m in (acc.split('-')[0] for x in acc[1:-1].split(',')) if UP.match(m)]
                    if mem:
                        groups.append(mem)
                else:
                    b = acc.split('-')[0]
                    if UP.match(b):
                        groups.append(b)
            if len(groups) >= 2:
                out['CP:' + r[iac]] = groups
    return out


def main():

    parser = argparse.ArgumentParser(description="Non negative matrix foactorization for fractionation data")
    parser.add_argument("--input_msds_pickle", action="store", dest="msds_filename", required=True, 
                                            help="Filename of MSDS pickle: pickle comes from running protein_complex_maps.util.read_ms_elutions_pickle_MSDS.py")
    parser.add_argument("--proteins", action="store", dest="proteins", nargs='+', required=False, 
                                            help="Protein ids in which to anaylze")
    parser.add_argument("--plot_filename", action="store", dest="plot_filename", required=False, default=None,
                                            help="Filename of output plot")
    parser.add_argument("--ignore_missing", action="store_true", dest="ignore_missing", required=False, default=False,
                                            help="Ignore missing protein ids in msds")
    parser.add_argument("--genenames", action="store_true", dest="genenames", required=False, default=False,
                                            help="Set labels to genenames")
    parser.add_argument("--rank", action="store", type=int, dest="rank", required=False, default=2,
                                            help="Rank or number of clusters")
    parser.add_argument("--max_iter", action="store", type=int, dest="max_iter", required=False, default=30,
                                            help="Maximum number of iterations for factorization")
    parser.add_argument("--threshold", action="store", type=float, dest="threshold", required=False, default=0.0,
                                            help="Coefficient threshold for membership in a bicluster, default 0.0")
    
    # Load in ComplexPortal complexes as a list of lists
    reference_dict = load_complex_portal("9606.tsv")
    reference_complexes = [cpx_members for cpx, cpx_members in reference_dict.items()]

    args = parser.parse_args()

    msds = pickle.load( open( args.msds_filename, "rb" ) )

    # Restrict data set to user-specified PDB accessions
    if args.proteins != None:
            data_set, new_id_map = msds.get_subdata_matrix(args.proteins, ignoreNonExistingIds=args.ignore_missing) 
    else:
            data_set = msds.get_data_matrix()
            new_id_map = msds.get_name2index()

    acc_map = dict(new_id_map)
    if args.genenames:
            print "new_id_map.values: %s" % (new_id_map.values(),)
            genename_map = pu.get_genenames_uniprot( new_id_map.values() )
            #print new_id_map
            #print genename_map
            gene_id_map = dict()
            for i in xrange(len(data_set)):
                    #print i
                    #print genename_map[new_id_map[i]]
                    #kdrew: sometimes no genename is returned for certain ids, default to original id
                    try:
                            if genename_map[new_id_map[i]] == None:
                                    gene_id_map[i] = new_id_map[i]
                            else:
                                    gene_id_map[i] = genename_map[new_id_map[i]]
                    except KeyError:
                            gene_id_map[i] = new_id_map[i]

            new_id_map = gene_id_map
    #logging.debug(clean_data_matrix)
    #kdrew: make all rows sum to 1.0
    #data_set = nu.normalize_over_columns(data_set)
    #kdrew: make all columns sum to 1.0
    data_set_raw = data_set
    data_set = nu.normalize_over_rows(data_set)

    V = data_set
    V_raw = data_set_raw

    print V

    # Run LSNMF algorithm
    # Returned object is fitted factorization model. Through it user can access quality and performance measures.
    # The fctr_res's attribute `fit` contains all the attributes of the factorization.  
    #fctr = nimfa.mf(V, method = "lsnmf", max_iter = args.max_iter, rank = args.rank)
    #fctr_res = nimfa.mf_run(fctr)

    lsnmf = nimfa.Lsnmf(V, max_iter=args.max_iter, rank=args.rank)
    fctr_res = lsnmf()

    lsnmf_raw = nimfa.Lsnmf(V_raw, max_iter=args.max_iter, rank=args.rank)
    fctr_res_raw = lsnmf_raw()

    # Basis matrix.
    W = fctr_res.basis()
    print "Basis matrix"
    print W

    # Mixture matrix. 
    H = fctr_res.coef()
    print "Coef"
    print H

    # Basis matrix for non-normalized
    W_raw = fctr_res_raw.basis()
    
    # Mixture matrix for non-normalized
    H_raw = fctr_res_raw.coef()

    # Print the loss function according to Kullback-Leibler divergence. By default Euclidean metric is used.
    print "Distance Kullback-Leibler: %5.3e" % fctr_res.distance(metric = "kl")

    # Compute generic set of measures to evaluate the quality of the factorization
    sm = fctr_res.summary()
    # Print residual sum of squares (Hutchins, 2008). Can be used for estimating optimal factorization rank.
    print "Rss: %8.3f" % sm['rss']
    # Print explained variance.
    print "Evar: %8.3f" % sm['evar']
    # Print actual number of iterations performed
    print "Iterations: %d" % sm['n_iter']

    # Print estimate of target matrix V 
    #print "Estimate"
    #print np.dot(W, H)

    # Obtain a random sampling score object using the data_set (V), and a fraction-wise dot multiplication of abundances
    #   - This product is large only if every protein in the group has a high value in that column
    #   - A single protein near zero collapses this product
    rsscore_obj = rsu.RandomSamplingScore(data_set, su.multiple_dot, sample_module=np.random)
    #score_function = rsscore_obj.zscore_all(matrix=data_set, mode='two_sided')
    score_function = rsscore_obj.zscore_all(matrix=data_set, mode='one_sided')

    # Do the same for non-normalized data
    rsscore_obj_raw = rsu.RandomSamplingScore(data_set_raw, su.multiple_dot, sample_module=np.random)
    score_function = rsscore_obj.zscore_all(matrix=data_set_raw, mode='one_sided')

    # Initialize BiclusterGenerator object with (score_function, num_terations, starting_temp, random_sampler)
    bcgen = bg.BiclusterGenerator(rsscore_obj.zscore_all_neg, iterations=5, starting_temperature = 0.00000001, random_module=np.random)

    # Do the same for non-normalized data
    bcgen_raw = bg.BiclusterGenerator(rsscore_obj_raw.zscore_all_neg, iterations=5, starting_temperature = 0.00000001, random_module=np.random)

    # Build the RateQuenchAnnealer object around the Monte Carlo object created by the previous line. Attach it to bcgen
    ratequench_annealer = anl.RateQuenchAnnealer( bcgen.get_montecarlo(), quench_iteration=4, scale=0.9, rate=0.3, recent_iterations=10, adjust_scale=True )
    bcgen.set_annealer(ratequench_annealer)

    # Do the same for non-normalized data
    ratequench_annealer_raw = anl.RateQuenchAnnealer( bcgen_raw.get_montecarlo(), quench_iteration=4, scale=0.9, rate=0.3, recent_iterations=10, adjust_scale=True )
    bcgen_raw.set_annealer(ratequench_annealer_raw)


    #bicluster1 = bc.Bicluster(rows=[0,1,2,3,4,5,14], cols=[55,56], random_module=random)
    #print "lid zscore: %s" % (rsscore_obj.zscore_all(bicluster1.get_submatrix(data_set)),)
    #print "lid cols zscore: %s" % (rsscore_obj.zscore_columns(data_set, bicluster1))
    #print "lid rows zscore: %s" % (rsscore_obj.zscore_rows(data_set, bicluster1))

    print "Clusters"
    ref_names, ref = bench.prep_reference_named(reference_dict, set(acc_map.values()))
    stat_rows = []
    stat_rows_raw = []

    nmf_clusters = []
    mc_clusters = []
    mc_scores = []

    nmf_clusters_raw = []
    mc_clusters_raw = []
    mc_scores_raw = []
    for i in range(args.rank):
        Wtransvec = np.transpose(W)[i]
        Wtransvec_raw = np.transpose(W_raw)[i]
        #print "W",i, Wtransvec
        rows = np.ravel(np.where(Wtransvec > args.threshold)[1])
        rows_raw = np.ravel(np.where(Wtransvec_raw > args.threshold)[1])

        Hvec = H[i]
        Hvec_raw = H_raw[i]
        #print "H",i, Hvec
        cols = np.ravel(np.where(Hvec > args.threshold)[1])
        cols_raw = np.ravel(np.where(Hvec_raw > args.threshold)[1])

        #print rows
        #print cols
        bicluster1 = bc.Bicluster(rows=rows, cols=cols, random_module=random)
        bicluster1_raw = bc.Bicluster(rows=rows_raw, cols=cols_raw, random_module=random)

        nmf_clusters.append(set(acc_map[int(r)] for r in bicluster1.rows()))
        nmf_clusters_raw.append(set(acc_map[int(r)] for r in bicluster1_raw.rows()))

        pbc.plot_bicluster_worker( data_set, bicluster1, savefilename="bicluster_test%s.pdf" % (i,), ylim_max=True)
        plt.close('all')

        pbc.plot_bicluster_worker( data_set_raw, bicluster1_raw, savefilename="bicluster_raw_test%s.pdf" % (i,), ylim_max=True)
        plt.close('all')

        #kdrew: monte carlo optimize using nnmf cluster as seed
        bicluster_mc = bcgen.generator(data_set, bicluster1)
        bicluster_mc_raw = bcgen_raw.generator(data_set_raw, bicluster1_raw)

        mc_clusters.append(set(acc_map[int(r)] for r in bicluster_mc.rows()))
        mc_clusters_raw.append(set(acc_map[int(r)] for r in bicluster_mc_raw.rows()))

	eval_dict = bcgen.evaluate( data_set, len(bcgen.biclusters)-1 )
        eval_dict_raw = bcgen_raw.evaluate( data_set_raw, len(bcgen_raw.biclusters)-1 )

        mc_scores.append(eval_dict['all']['zscore'])
        mc_scores_raw.append(eval_dict_raw['all']['zscore'])

        # Plot and record evaluation
        nmf_set = set(acc_map[int(r)] for r in bicluster1.rows())
        nmf_set_raw = set(acc_map[int(r)] for r in bicluster1_raw.rows())

        mc_set = set(acc_map[int(r)] for r in bicluster_mc.rows())
        mc_set_raw = set(acc_map[int(r)] for r in bicluster_mc_raw.rows())

        st_nmf = bench.cluster_stats(nmf_set, ref, ref_names)
        st_nmf_raw = bench.cluster_stats(nmf_set_raw, ref, ref_names)

        st_mc = bench.cluster_stats(mc_set, ref, ref_names)
        st_mc_raw = bench.cluster_stats(mc_set_raw, ref, ref_names)

        bench.plot_cluster_stats(st_nmf, st_mc, "bicluster_testStats%s.pdf" % (i,), "component %s" %i, stats_nmf_raw=st_nmf_raw, stats_mc_raw=st_mc_raw)
        stat_rows += [(i, "nmf", st_nmf), (i, "mc", st_mc)]
        #stat_rows_raw += [(i, "nmf_raw", st_nmf_raw), (i, "mc_raw", st_mc_raw)]
        stat_rows += [(i, "nmf_raw", st_nmf_raw), (i, "mc_raw", st_mc_raw)]

        try:
            print "zscore: %s" % (rsscore_obj.zscore_all(bicluster1.get_submatrix(data_set)),)
            print "cols zscore: %s" % (rsscore_obj.zscore_columns(data_set, bicluster1))
            print "rows zscore: %s" % (rsscore_obj.zscore_rows(data_set, bicluster1))
        except:
            continue

        bench.write_stats_table(stat_rows, "bicluster_stats.tsv")
        #bench.write_stats_table(stat_rows_raw, "bicluster_raw_stats.tsv")

	for t in eval_dict.keys():
            logging.info("random %s, mean: %s, std: %s, zscore: %s" % ( t, eval_dict[t]['mean'], eval_dict[t]['std'], eval_dict[t]['zscore'] ))

	pbc.plot_bicluster_worker(data_set, bicluster_mc, savefilename="bicluster_testMC%s.pdf" % (i), ylim_max=True)
        plt.close('all')

	pbc.plot_bicluster_worker(data_set_raw, bicluster_mc_raw, savefilename="bicluster_raw_testMC%s.pdf" % (i), ylim_max=True)
        plt.close('all')

    detected = set(acc_map.values())
    #reference_complexes = load_complex_portal("9606.tsv")
    #print(reference_complexes)
    ref = bench.prep_reference(reference_complexes, detected)

    for name, clusters in [("NMF only", nmf_clusters), ("NMF only (raw counts)", nmf_clusters_raw), ("NMF + MC", mc_clusters), ("NMF + MC (raw counts)", mc_clusters_raw)]:
        cp, cr = bench.complex_level(clusters, ref, thr=0.25)
        pp, pr = bench.pairwise(clusters, ref)
        print "%s complex P/R: %.2f/%.2f  pairwise P/R: %.2f/%.2f" % (name, cp, cr, pp, pr)

    scores = bench.scores_from_clusters(mc_clusters, mc_scores)
    (prec, rec, thr), ap = bench.pair_curve(scores, ref)

    scores_raw = bench.scores_from_clusters(mc_clusters_raw, mc_scores_raw)
    (prec_raw, rec_raw, thr), ap_raw = bench.pair_curve(scores_raw, ref)

    print "pairwise average precision %.3f" % ap

    data_subplots = []
    f, data_subplots = plt.subplots(len(W),1,sharex='col')

    for i, data_r in enumerate(W):
        #print "pos: %s name: %s" % (i, new_id_map[i])
        data_row = np.array(data_r.reshape(-1))[0]
        barcolor = "gray"

        data_subplots[i].bar(np.arange(len(data_row)), data_row, align='center', facecolor=barcolor)
        data_subplots[i].axes.set_yticklabels([],visible=False)
        data_subplots[i].set_ylabel("%s" % (new_id_map[i],),rotation='horizontal', color=barcolor, fontsize=6, verticalalignment="center",horizontalalignment="right")


    for i, data_r in enumerate(W_raw):
        #print "pos: %s name: %s" % (i, new_id_map[i])
        data_row = np.array(data_r.reshape(-1))[0]
        barcolor = "green"

        data_subplots[i].bar(np.arange(len(data_row)), data_row, align='center', facecolor=barcolor)
        data_subplots[i].axes.set_yticklabels([],visible=False)
        data_subplots[i].set_ylabel("%s" % (new_id_map[i],),rotation='horizontal', color=barcolor, fontsize=6, verticalalignment="center",horizontalalignment="right")

    plt.savefig(args.plot_filename)


if __name__ == "__main__":
	main()

