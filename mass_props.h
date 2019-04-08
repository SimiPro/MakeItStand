
using namespace Eigen;

double getMass(const Eigen::VectorXd &s10) {
    return s10[0];
}

Vector3d getCoM(const Eigen::VectorXd &s10) {
    double M = getMass(s10);
    Vector3d result;
    result << s10[1], s10[2], s10[3];
    if (M == 0) return Vector3d(0,0,0);
    return result / M;
}


void props(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &FN, const double phi,  Eigen::VectorXd &s10) {
    // triangle vertices counter clockwise: a, b, c
    s10.resize(10); s10.setZero();

    for (int i = 0; i < F.rows(); i++) {
        Vector3d a = V.row(F.row(i)[0]).transpose();
        Vector3d b = V.row(F.row(i)[1]).transpose();
        Vector3d c = V.row(F.row(i)[2]).transpose();

        Vector3d u = (b - a), v = (c - a);
        Vector3d n = u.cross(v);

        Vector3d h1 = a + b + c;
        Vector3d h2 = a.array()*a.array() + b.array()*(a + b).array();
        Vector3d h3 = h2.array() + c.array()*h1.array();
        Vector3d h4 = a.array()*a.array()*a.array() + b.array()*h2.array() + c.array()*h3.array();
        Vector3d h5 = h3.array() + a.array()*(h1.array() + a.array());
        Vector3d h6 = h3.array() + b.array()*(h1.array() + b.array());
        Vector3d h7 = h3.array() + c.array()*(h1.array() + c.array());
        Vector3d a_bar; a_bar << a[1], a[2], a[0];
        Vector3d b_bar; b_bar << b[1], b[2], b[0];
        Vector3d c_bar; c_bar << c[1], c[2], c[0];

        Vector3d h8 = a_bar.array()*h5.array() + b.array()*h6.array() + c.array()*h7.array();
        s10[0] += (n.array()*h1.array())[0];

        Vector3d stemp = n.array()*h3.array();
        s10[1] += stemp[0]; s10[2] += stemp[1]; s10[3] += stemp[2];

        stemp = n.array()*h8.array();
        s10[4] += stemp[0]; s10[5] += stemp[1]; s10[6] += stemp[2];

        stemp = n.array()*h4.array();
        s10[7] += stemp[0]; s10[8] += stemp[1]; s10[9] += stemp[2];
    }

    s10[0] *= 1./6.;
    s10[1] *= 1./24.; s10[2] *= 1./24.; s10[3] *= 1./24.;
    s10[4] *= 1./120.; s10[5] *= 1./120.; s10[6] *= 1./120.;
    s10[7] *= 1./60.; s10[8] *= 1./60.; s10[9] *= 1./60.;

    s10 *= phi;
}