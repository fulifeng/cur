#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
try:
    import numpy
    from numpy import linalg as LA
except:
    print "This implementation requires the numpy module."
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, missing_denotation=-1):
    log_sl = steps / 10
    last_e = 0
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                # if R[i][j] > 0:
                if abs(R[i][j] - missing_denotation) > 1e-8:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                # if R[i][j] > 0:
                if abs(R[i][j] - missing_denotation) > 1e-8:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if step % log_sl == log_sl - 1:
            print 'current error:', e
        if abs(e - last_e) < 0.001:
            break
        last_e = e
    return P, Q.T


def get_missing_entry(R, k, steps=5000, alpha=0.0002, beta=0.02, missing_denotation=-1):
    n = len(R)
    m = len(R[0])
    P = numpy.random.rand(n, k)
    Q = numpy.random.rand(m, k)
    nP, nQ = matrix_factorization(R, P, Q, k, steps=steps, alpha=alpha, beta=beta)
    nR = numpy.dot(nP, nQ.T)
    # cR = numpy.ones((n,m))
    for i in xrange(n):
        for j in xrange(m):
            # if R[i][j] > 0:
            if abs(R[i][j] - missing_denotation) < 1e-8:
                # cR[i, j] = nR[i, j]
                R[i][j] = nR[i, j]
            # else:
                # cR[i, j] = R[i, j]
    print 'final error', pow(LA.norm(R - nR), 2)
    return R


###############################################################################

if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = numpy.array(R, dtype=float)

    N = len(R)
    M = len(R[0])
    K = 2

    # P = numpy.random.rand(N,K)
    # Q = numpy.random.rand(M,K)
    #
    # nP, nQ = matrix_factorization(R, P, Q, K)
    # print numpy.dot(nP, nQ.T)

    print '-------------------------------'
    print get_missing_entry(R, K, missing_denotation=0)
    print R
