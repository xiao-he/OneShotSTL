package com.alibaba.OneShotSTL;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

/**
 * 2022/7/21
 *
 * @author xiao.hx@alibaba-inc.com
 */
public class OnlineBandedSystemAlg {
    private int p;
    private int n;
    private int m;
    private int pPlus1;
    private int pPlusN;

    private double[] bArray;
    private double[] DArray;
    private DMatrixRMaj LMatrix;

    private double[] LArrayTmp;
    private double[] DArrayTmp;
    private double[] bArrayTmp;
    private double[] bArrayTmpCopy;

    private CommonOps_DDRM ops = new CommonOps_DDRM();

    public OnlineBandedSystemAlg(int p) {
        this.p = p;
        this.pPlus1 = p + 1;
        this.m = p / 2;
        this.n = m * 3;
        this.pPlusN = p + n;
    }

    public void init(DMatrixRMaj AMatrix, DMatrixRMaj bMatrix) {
        double[][] LArray = new double[2 * p][2 * p];
        double[] DArrayTmp = new double[2 * p];
        double[] bArrayTmp = bMatrix.getData();

        SymmetricDoolittleInit(AMatrix.getData(), bArrayTmp, LArray, DArrayTmp);

        DMatrixRMaj LMatrixTmp = new DMatrixRMaj(LArray);
        LMatrix = new DMatrixRMaj(2 * p, p);
        ops.extract(LMatrixTmp,0, 2 * p, 0, p,  LMatrix);
        DArray = new double[p];
        System.arraycopy(DArrayTmp, 0, DArray, 0, p);
        bArray = new double[p];
        System.arraycopy(bArrayTmp, 0, bArray, 0, p);
    }

    public double[] onlineSolve(DMatrixRMaj A, DMatrixRMaj b, boolean updateModel) {
        DMatrixRMaj LMatrixTmp = new DMatrixRMaj(pPlusN, pPlusN);
        ops.insert(LMatrix, LMatrixTmp, 0, 0);
        DArrayTmp = new double[pPlusN];
        System.arraycopy(DArray, 0, DArrayTmp, 0, p);
        bArrayTmp = new double[pPlusN];
        System.arraycopy(bArray, 0, bArrayTmp, 0, p);
        System.arraycopy(b.getData(), 0, bArrayTmp, p, n);

        LArrayTmp = LMatrixTmp.getData();
        SymmetricDoolittleOnline(A.getData(), bArrayTmp, LArrayTmp, DArrayTmp);

        if (updateModel) {
            update();
        } else {
            bArrayTmpCopy = new double[pPlusN];
            System.arraycopy(bArrayTmp, 0, bArrayTmpCopy, 0, pPlusN);
        }

        backwardSubstitution(LArrayTmp, DArrayTmp, bArrayTmp);
        double[] xArray = new double[2 * m + p];
        System.arraycopy(bArrayTmp, bArrayTmp.length - 2 * m - p, xArray, 0, 2 * m + p);
        if (!updateModel) {
            System.arraycopy(bArrayTmpCopy, 0, bArrayTmp, 0, pPlusN);
        }

        return xArray;
    }

    public void update() {
        DMatrixRMaj LMatrixTmp = new DMatrixRMaj(LArrayTmp);
        LMatrixTmp.reshape(pPlusN, pPlusN, true);
        ops.extract(LMatrixTmp, m, pPlusN, m, m + p, LMatrix);
        System.arraycopy(DArrayTmp, m, DArray, 0, p);
        System.arraycopy(bArrayTmp, m, bArray, 0, p);
    }

    private void SymmetricDoolittleInit(double[] A, double[] b, double[][] L, double[] D) {
        for (int k = 0; k < b.length; k++) {
            L[k][k] = 1;
            for (int i = k - p; i < k; i++) {
                if (i >= 0) {
                    D[k] += D[i] * L[k][i] * L[k][i];
                }
            }
//            D[k] = A[k][k] - D[k];
            D[k] = A[k + k * b.length] - D[k];
            for (int j = k + 1; j < b.length; j++) {
                if (j < k + pPlus1) {
                    for (int i = k - p; i < k; i++) {
                        if (i >= 0) {
                            L[j][k] += L[j][i] * D[i] * L[k][i];
                        }
                    }
//                    L[j][k] = (A[j][k] - L[j][k]) / D[k];
                    L[j][k] = (A[k + j * b.length] - L[j][k]) / D[k];
                }
            }
        }
        for (int j = 0; j < b.length; j++) {
            for (int i = j + 1; i < b.length; i++) {
                if (i < j + pPlus1) {
                    b[i] = b[i] - L[i][j] * b[j];
                }
            }
        }
    }

    private void SymmetricDoolittleOnline(double[] A, double[] b, double[] L, double[] D) {
        for (int k = p; k < pPlusN; k++) {
//            L[k][k] = 1;
            L[k + k * pPlusN] = 1;
            for (int i = k - p; i < k; i++) {
                if (i >= 0) {
//                    D[k] += D[i] * L[k][i] * L[k][i];
                    D[k] += D[i] * L[i + k * pPlusN] * L[i + k * pPlusN];
                }
            }
            for (int i = k - p; i < p; i++) {
                if (i >= 0) {
//                    b[k] -= L[k][i] * b[i];
                    b[k] -= L[i + k * pPlusN] * b[i];
                }
            }
//            D[k] = A[k-p][k-p] - D[k];
            D[k] = A[k - p + (k - p) * n] - D[k];
            for (int j = k + 1; j < pPlusN; j++) {
                if (j < k + pPlus1) {
                    for (int i = k - p; i < k; i++) {
                        if (i >= 0) {
//                            L[j][k] += L[j][i] * D[i] * L[k][i];
                            L[k + j * pPlusN] += L[i + j * pPlusN] * D[i] * L[i + k * pPlusN];
                        }
                    }
//                    L[j][k] = (A[j-p][k-p] - L[j][k]) / D[k];
                    L[k + j * pPlusN] = (A[k - p + (j - p) * n] - L[k + j * pPlusN]) / D[k];
                }
            }
        }
        for (int j = p; j < pPlusN; j++) {
            for (int i = j + 1; i < pPlusN; i++) {
                if (i < j + pPlus1) {
//                    b[i] = b[i] - L[i][j] * b[j];
                    b[i] = b[i] - L[j + i * pPlusN] * b[j];
                }
            }
        }
    }

    private void backwardSubstitution(double[] L, double[] D, double[] b) {
        for (int j = pPlusN - 1; j >= p; j--) {
//        for (int j : js) {
            b[j] = b[j] / D[j];
            for (int i = j - p; i < j; i++) {
//                b[i] = b[i] - D[i] * L[j][i] * b[j];
                b[i] = b[i] - D[i] * L[i + j * pPlusN] * b[j];
            }
        }
    }
}
