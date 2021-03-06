#include <vector>

using namespace std;
using namespace Eigen;

// VOXALIZATION 
typedef vector<vector<vector<int > > > Grid;
typedef vector<vector<int>> VII;
typedef vector<int> VI;

class Voxalization {
    MatrixXd &V;
    MatrixXi &F;
    RowVector3d &com;
    int resolution;
    Vector3d m;
    Vector3d M;
    Grid sdf;
    double dx, dy, dz;


public:
    Voxalization(MatrixXd &V_, MatrixXi &F_, int resolution_, RowVector3d &com_): V(V_), F(F_), resolution(resolution_),
                com(com_), sdf(resolution_, VII(resolution_, VI(resolution_, 0))) {

        // BOUNDING BOX
        m = V.colwise().minCoeff();
        M = V.colwise().maxCoeff();

        dx = (M(0) - m(0)) / resolution;
        dy = (M(1) - m(1)) / resolution;
        dz = (M(2) - m(2)) / resolution;

        for (int x = 0; x < resolution; x++) {
            for (int y = 0; y < resolution; y++) {
                for (int z = 0; z < resolution; z++) {
                    Vector3d box_center(m(0) + (x + 0.5)*dx, m(1) + (y + 0.5)*dy, m(2) + (z + 0.5)*dz);
                    if (is_in_mesh(box_center)) {
                        sdf[x][y][z] = -1;
                    } else {
                        sdf[x][y][z] = 1;
                    }
                }
            }
        }
    }
  

    bool is_in_mesh(const Vector3d &p) {
        Vector3d dir = com.transpose() - p;
        vector<igl::Hit> hits;
        if(ray_mesh_intersect(p, dir, V, F, hits)) {
            if (hits.size() % 2 == 1) return true;
        }
        return false;
    }


    void triangulate(Eigen::MatrixXd &new_V, Eigen::MatrixXi &new_F, MatrixXd &new_N) {
        MatrixXd tmpV;
        MatrixXi tmpF;
        MatrixXd tmpN;
        tmpV.resize(resolution*resolution*resolution*8, 3);
        tmpF.resize(resolution*resolution*resolution*12, 3);
        tmpN.resize(resolution*resolution*resolution*12, 3);
        int v_counter = 0, f_counter = 0;
        std::cout << "triangulation progress: " << endl;
        for (int x = 0; x < resolution; x++) {
            for (int y = 0; y < resolution; y++) {
                for (int z = 0; z < resolution; z++) {
                    int idx = x*resolution*resolution+y*resolution+z;
                    if (idx % 1000 == 0) {
                        float progress = (100./(resolution*resolution*resolution)*(idx));
                        cout << progress << "% " << endl << flush;
                    }
                    

                    if (sdf[x][y][z] >= 0) continue; 
                    int v_start = v_counter;

                    RowVector3d v0(m(0) + x*dx, m(1) + y*dy, m(2) + z*dz);
                    tmpV.row(v_counter++) = v0; // 0
                    tmpV.row(v_counter++) = v0 + RowVector3d(dx, 0, 0); // 1
                    tmpV.row(v_counter++) = v0 + RowVector3d(0, dy, 0); // 2
                    tmpV.row(v_counter++) = v0 + RowVector3d(dx, dy, 0); // 3
                    tmpV.row(v_counter++) = v0 + RowVector3d(dx, 0, dz); // 4
                    tmpV.row(v_counter++) = v0 + RowVector3d(dx, dy, dz); // 5
                    tmpV.row(v_counter++) = v0 + RowVector3d(0, 0, dz); // 6
                    tmpV.row(v_counter++) = v0 + RowVector3d(0, dy, dz); // 7

                    // front
                    tmpF.row(f_counter++) = RowVector3i(v_start + 0, v_start + 1, v_start + 2);
                    tmpN.row(f_counter-1) = RowVector3d(0,0,-1);
                    tmpF.row(f_counter++) = RowVector3i(v_start + 1, v_start + 3, v_start + 2);
                    tmpN.row(f_counter-1) = RowVector3d(0,0,-1);

                    // right
                    tmpF.row(f_counter++) = RowVector3i(v_start + 1, v_start + 4, v_start + 3);
                    tmpN.row(f_counter-1) = RowVector3d(1,0,0);
                    tmpF.row(f_counter++) = RowVector3i(v_start + 4, v_start + 5, v_start + 3);
                    tmpN.row(f_counter-1) = RowVector3d(1,0,0);


                    // bottom 
                    tmpF.row(f_counter++) = RowVector3i(v_start + 0, v_start + 1, v_start + 6);
                    tmpN.row(f_counter-1) = RowVector3d(0,-1,0);
                    tmpF.row(f_counter++) = RowVector3i(v_start + 1, v_start + 4, v_start + 6);
                    tmpN.row(f_counter-1) = RowVector3d(0,-1,0);

                    // left
                    tmpF.row(f_counter++) = RowVector3i(v_start + 0, v_start + 6, v_start + 2);
                    tmpN.row(f_counter-1) = RowVector3d(-1,0,0);
                    tmpF.row(f_counter++) = RowVector3i(v_start + 6, v_start + 7, v_start + 2);
                    tmpN.row(f_counter-1) = RowVector3d(-1,0,0);

                    // top
                    tmpF.row(f_counter++) = RowVector3i(v_start + 5, v_start + 7, v_start + 2);
                    tmpN.row(f_counter-1) = RowVector3d(0,1,0);
                    tmpF.row(f_counter++) = RowVector3i(v_start + 3, v_start + 5, v_start + 2);
                    tmpN.row(f_counter-1) = RowVector3d(0,1,0);

                    // back
                    tmpF.row(f_counter++) = RowVector3i(v_start + 6, v_start + 5, v_start + 7);
                    tmpN.row(f_counter-1) = RowVector3d(0,0,1);
                    tmpF.row(f_counter++) = RowVector3i(v_start + 6, v_start + 4, v_start + 5);
                    tmpN.row(f_counter-1) = RowVector3d(0,0,1);
                }
            }
        }
        cout << endl << "finihsed triangulation" << endl;

        new_V = tmpV.block(0,0, v_counter, 3);
        new_F = tmpF.block(0,0, f_counter, 3);
        new_N = tmpN.block(0,0, f_counter, 3);
    }
    

};



// END VOXALIZATION

