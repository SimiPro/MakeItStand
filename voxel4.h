#include <vector>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/matrix_to_list.h>
#include <igl/barycenter.h>

using namespace std;
using namespace Eigen;

class Voxalization {
public:
    const MatrixXd &V;
    const MatrixXi &F;

    MatrixXd TV; // #V by 3
    MatrixXi TF; // tet face indices #F by 3
    MatrixXi TT; // triangle face indices #T by 4 

    Eigen::MatrixXd B;

    vector<bool> empty;


    Voxalization(MatrixXd &V_, MatrixXi &F_, int resolution_, int max_depth_): V(V_), F(F_) {

        cout << "Start tetrahedralize" << endl;

        int success = igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.414a0.01" , TV, TT, TF); // or use pq1.414Y?
        if (0 == success) {
            cout << "Successfully tetrahedralized mesh" << endl;
            cout << "Num Vertices: " << TV.rows() << endl;
            cout << "Num Tets: " << TT.rows() << endl;
            cout << "Num Faces: " << TF.rows() << endl;
            empty.resize(TT.rows());
            fill(empty.begin(), empty.end(), false);
            // Compute barycenters
            igl::barycenter(TV,TT,B);
        }
        if (1 == success)
            cout << "tetrahedralization failed. " << endl;
        if (2 == success)
            cout << "Tetra did not crash but do you have holes in ur mesh???" << endl;

        // V_ is now V_ outer
        MatrixXd VN; 
        igl::per_vertex_normals(V, F, VN);
        VN.rowwise().normalize();
        V_ = V + VN*0.01; 

    }

    void empty_box(int id) {
        empty[id] = true;
    }


    void get_interior_mesh(Eigen::MatrixXd &new_V, vector<MatrixXi> &faces) {
        MatrixXd V_temp(TT.rows()*4,3);

        for (int i = 0; i < TT.rows(); i++) {
            if (empty[i]) continue;
            V_temp.row(i*4+0) = TV.row(TT(i, 0));
            V_temp.row(i*4+1) = TV.row(TT(i, 1));
            V_temp.row(i*4+2) = TV.row(TT(i, 2));
            V_temp.row(i*4+3) = TV.row(TT(i, 3));
            
            MatrixXi tmpFace; tmpFace.resize(4, 3);
            tmpFace.row(0) << (i*4)+0, (i*4)+1, (i*4)+3;
            tmpFace.row(1) << (i*4)+0, (i*4)+2, (i*4)+1;
            tmpFace.row(2) << (i*4)+3, (i*4)+2, (i*4)+0;
            tmpFace.row(3) << (i*4)+1, (i*4)+2, (i*4)+3;
            faces.push_back(tmpFace);
        }
        new_V = V_temp;
    }

    void triangulate_empty(Eigen::MatrixXd &new_V, Eigen::MatrixXi &new_F) {
        vector<MatrixXi> faces;
        MatrixXd V_temp(TT.rows()*4,3);

        for (int i = 0; i < TT.rows(); i++) {
            if (!empty[i]) continue;
            V_temp.row(i*4+0) = TV.row(TT(i, 0));
            V_temp.row(i*4+1) = TV.row(TT(i, 1));
            V_temp.row(i*4+2) = TV.row(TT(i, 2));
            V_temp.row(i*4+3) = TV.row(TT(i, 3));
            
            MatrixXi tmpFace; tmpFace.resize(4, 3);
            tmpFace.row(0) << (i*4)+0, (i*4)+1, (i*4)+3;
            tmpFace.row(1) << (i*4)+0, (i*4)+2, (i*4)+1;
            tmpFace.row(2) << (i*4)+3, (i*4)+2, (i*4)+0;
            tmpFace.row(3) << (i*4)+1, (i*4)+2, (i*4)+3;
            faces.push_back(tmpFace);
        }
        
        new_V = V_temp;
        new_F.resize(4*faces.size(), 3);

        for (int i = 0; i < faces.size(); i++) {
            new_F.row(i*4 + 0) = faces[i].row(0);
            new_F.row(i*4 + 1) = faces[i].row(1);
            new_F.row(i*4 + 2) = faces[i].row(2);
            new_F.row(i*4 + 3) = faces[i].row(3);
        }
    }

    void triangulate(Eigen::MatrixXd &new_V, Eigen::MatrixXi &new_F) {
        vector<MatrixXi> faces;
        get_interior_mesh(new_V, faces);
        new_F.resize(4*faces.size(), 3);

        for (int i = 0; i < faces.size(); i++) {
            new_F.row(i*4 + 0) = faces[i].row(0);
            new_F.row(i*4 + 1) = faces[i].row(1);
            new_F.row(i*4 + 2) = faces[i].row(2);
            new_F.row(i*4 + 3) = faces[i].row(3);
        }
    }


    void triangulate1(Eigen::MatrixXd &new_V, Eigen::MatrixXi &new_F) {
        double t = 1;

        VectorXd v = B.col(2).array() - B.col(2).minCoeff();
        v /= v.col(0).maxCoeff();

        vector<int> s;

        for (unsigned i=0; i < v.size(); ++i)
          if (v(i) < t)
            s.push_back(i);

        new_V.resize(s.size()*4, 3);
        new_F.resize(s.size()*4, 3);

        for (unsigned i=0; i< s.size();++i) {
          new_V.row(i*4+0) = TV.row(TT(s[i],0));
          new_V.row(i*4+1) = TV.row(TT(s[i],1));
          new_V.row(i*4+2) = TV.row(TT(s[i],2));
          new_V.row(i*4+3) = TV.row(TT(s[i],3));
          new_F.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
          new_F.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
          new_F.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
          new_F.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
        }
    }

    void marching_cubes(const MatrixXd &V_voxels, const MatrixXi &F_voxels, const int resolution,
                 MatrixXd &SV, MatrixXi &SF) {}
};



// END VOXALIZATION

