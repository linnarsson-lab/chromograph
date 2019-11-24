def strFrags_to_list(frags):
    '''
    '''
    frags = frags.replace('[', '')
    frags = frags.replace(']', '')
    frags = frags.replace('"', '')
    frags = frags.replace("'", '')
    frags = frags.replace(' ', '')
    frags = frags.split(',')
    frag_list = [[frags[3*i], int(frags[3*i+1]), int(frags[3*i+2])]for i in range(int(len(frags)/3))]
    return frag_list