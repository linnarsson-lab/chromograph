import os
import numpy as np
import tempfile
import shutil
import chromograph
from chromograph.pipeline.utils import div0


def Generate_homer_motifs(TF_names: str, ref_pfm: str):
    """
    Generate a .motifs refence from a

    Args:
        ds                    Loom connection

    Remarks:

    """

    out_dir = os.path.join(chromograph.__path__[0], 'references', 'human_TFs.motifs')
    logging.info("Generating homer motifs, saving in chromograph reference directory")

    pfm_dir = tempfile.mkdtemp()

    ## Load data from .pfm reference and split into seperate files
    with open(ref_pfm) as f:
        i = 0
        valid = 0
        for line in f:

            if i%5 == 0:
                TF = line.split('_')[0][1:]
                ## If TF is in list of valids
                if TF in TF_names:
                    valid = 1
                    f_pmf = os.path.join(pfm_dir, f'{TF}.pfm')
                    f = open(f_pmf, "w")
                    f.write(line)
                else:
                    valid = 0
            else:
                if valid:
                    f.write(line)

                    if i%5 == 4:
                        f.close    
            i += 1
    logging.info('Saved pfms to temporary folder')
    logging.info('Calculating expected threshold')
    
    ## Calculate the maximum score (perfect match) and average score with 1 substitution
    mmax = []
    msub = []

    ## Load files and convert to numpy arrays
    for fpfm in [os.path.join(pfm_dir, x) for x in os.listdir(pfm_dir)]:
        with open(fpfm) as f:
            i = 0
            motif = []
            for line in f:
                if i%5 == 0:
                    TF = line.split('_')[0][1:]
                else:
                    line = [x for x in line.split(' ') if x is not ""]
                    motif.append(line[2:-2])

                    if i%5 == 4:
                        f.close    
                i += 1
        motif = np.array(motif).astype(int).T
        motif = np.around(div0(motif, np.sum(motif, axis = 1).reshape(motif.shape[0],1)), decimals=3) ## Convert to PPM

        ## Get maximum score for PPM
        mscore = 0
        for x in range(motif.shape[0]):
            mscore += np.log((np.max(motif[x,:])+0.001) / .25)
        mmax.append(np.around(mscore, decimals = 3))

        ## Get scores for single substitutions to 2nd highest scoring value
        sub_scores = []
        for i in range(motif.shape[0]):
            score = 0
            for x in range(motif.shape[0]):
                if x != i:
                    score += np.log((np.max(motif[x,:])+0.001) / .25)
                if x == i:
                    score += np.log((np.partition(motif[x,:], -2)[-2]+0.001) / .25)
            sub_scores.append(score)

        msub.append(np.around(np.mean(sub_scores), decimals=3))

    # calc the trendline between maximum and substitution scores
    z = np.polyfit(mmax, msub, 1)
    p = np.poly1d(z)

    logging.info('Saving motifs to chromograph reference')
    ## Reload all motifs and save in correct format
    with open(out_dir, 'w') as fmotif:
        for fpfm in [os.path.join(pfm_dir, x) for x in os.listdir(pfm_dir)]:
            with open(fpfm) as f:
                i = 0
                motif = []
                for line in f:
                    if i%5 == 0:
                        TF = line.split('_')[0][1:]
                    else:
                        line = [x for x in line.split(' ') if x is not ""]
                        motif.append(line[2:-2])

                        if i%5 == 4:
                            f.close    
                    i += 1
            motif = np.array(motif).astype(int).T
            motif = np.around(div0(motif, np.sum(motif, axis = 1).reshape(motif.shape[0],1)), decimals=3)

            ## Get maximum score
            score = 0
            for x in range(motif.shape[0]):
                score += np.log((np.max(motif[x,:])+0.001) / .25)
                
            ## Calculate expected threshold
            thres = p(score)
    
            ## Write header as >NAME NAME threshold
            header = [f'>{TF}', TF, str(thres)]
            fmotif.write('\t'.join(header) + '\n')

            ## Write PPM to file
            for line in motif.tolist():
                line = [str(x) for x in line]
                fmotif.write('\t'.join(line) + '\n')

        ## Close file
        fmotif.close()

    logging.info('Deleting tempdir')
    
    ## Finish by removing tempdir
    shutil.rmtree(pfm_dir)
    
    return 