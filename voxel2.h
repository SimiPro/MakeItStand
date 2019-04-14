#include <vector>

using namespace std;
using namespace Eigen;


// VOXALIZATION 

struct Box {
    bool filled;
    double sdf;
    bool is_boundary;
    Vector3d center;

};

struct Tripletz {
    int x,y,z;
};


typedef vector<vector<vector<Box > > > Grid;
typedef vector<vector<Box>> VII;
typedef vector<Box> VI;



class Voxalization {
    MatrixXd &V;
    MatrixXi &F;
    RowVector3d &com;
    int resolution;
    Vector3d m;
    Vector3d M;
    Grid grid;
    double dx, dy, dz;
    vector<Tripletz> box_id_to_grid_id; 

public:
    Voxalization(MatrixXd &V_, MatrixXi &F_, int resolution_, RowVector3d &com_): V(V_), F(F_), resolution(resolution_),
                com(com_), grid(resolution_, VII(resolution_, VI(resolution_, {true, 0, false}))) {

        // BOUNDING BOX
        m = V.colwise().minCoeff();
        M = V.colwise().maxCoeff();

        dx = (M(0) - m(0)) / resolution;
        dy = (M(1) - m(1)) / resolution;
        dz = (M(2) - m(2)) / resolution;

        
        // BUILD QUERY POINTS
        Eigen::MatrixXd Q;
        Q.resize(resolution*resolution*resolution, 3);


        int counter = 0;
        for (int x = 0; x < resolution; x++) {
            for (int y = 0; y < resolution; y++) {
                for (int z = 0; z < resolution; z++) {
                    Vector3d box_center(m(0) + (x + 0.5)*dx, m(1) + (y + 0.5)*dy, m(2) + (z + 0.5)*dz);
                    Q.row(counter++) = box_center;
                    grid[x][y][z].center = box_center;
                }
            }
        }
        // 

        // Ask that query
        VectorXd S, I;
        MatrixXd C, N;

        igl::SignedDistanceType type = igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_PSEUDONORMAL;
        igl::signed_distance(Q, V, F, type,
            numeric_limits<float>::min(), numeric_limits<float>::max(), S, I, C, N);


        // set in/out
        counter = 0;
        int boundary_counter = 0;
        for (int x = 0; x < resolution; x++) {
            for (int y = 0; y < resolution; y++) {
                for (int z = 0; z < resolution; z++) {
                    grid[x][y][z].sdf = S[counter++];
                    if (pow(grid[x][y][z].sdf,2) <= pow(dx/2, 2) + pow(dy/2, 2) + pow(dz/2, 2)) {
                        grid[x][y][z].is_boundary = true;
                        boundary_counter++;
                    }
                }
            }
        }

        cout << "Of " << counter << " boxes are " << boundary_counter << " boundary boxes" << endl;
    }

    void empty_box(int id) {
        assert(id < box_id_to_grid_id.size() && id >= 0);
        Tripletz trip = box_id_to_grid_id[id];
        grid[trip.x][trip.y][trip.z].filled = false;

    }


    void get_interior_mesh(Eigen::MatrixXd &new_V, vector<MatrixXi> &faces) {
        MatrixXd tmpV;
        tmpV.resize(resolution*resolution*resolution*8, 3);

        int v_counter = 0;
        std::cout << "triangulation progress: " << endl;
        int filled_boxes = 0;
        for (int x = 0; x < resolution; x++) {
            for (int y = 0; y < resolution; y++) {
                for (int z = 0; z < resolution; z++) {
                    if (grid[x][y][z].sdf >= 0 || !grid[x][y][z].filled) continue;
                    if (grid[x][y][z].is_boundary) continue;

                    box_id_to_grid_id.push_back({x,y,z}); filled_boxes++;
                    int v_start = v_counter;
                    RowVector3d v0(m(0) + x*dx, m(1) + y*dy, m(2) + z*dz);
                    addBoxPoints(tmpV, v_counter, v0);
                    MatrixXi currFace; currFace.resize(12, 3);
                    int vs = 0;
                    addFaces(currFace, vs, v_start);

                    faces.push_back(currFace);
                }
            }
        }

        new_V = tmpV.block(0,0, v_counter, 3);

        std::cout << "triangulation process finished with : " << filled_boxes << " filled boxes" << std::endl;

    }

    void addBoxPoints(MatrixXd &tmpV, int &v_counter, const RowVector3d &v0) {
        tmpV.row(v_counter++) = v0; // 0
        tmpV.row(v_counter++) = v0 + RowVector3d(dx, 0, 0); // 1
        tmpV.row(v_counter++) = v0 + RowVector3d(0, dy, 0); // 2
        tmpV.row(v_counter++) = v0 + RowVector3d(dx, dy, 0); // 3
        tmpV.row(v_counter++) = v0 + RowVector3d(dx, 0, dz); // 4
        tmpV.row(v_counter++) = v0 + RowVector3d(dx, dy, dz); // 5
        tmpV.row(v_counter++) = v0 + RowVector3d(0, 0, dz); // 6
        tmpV.row(v_counter++) = v0 + RowVector3d(0, dy, dz); // 7
    }


    void addFaces(MatrixXi &tmpF, int &f_counter, int v_start) {
        // front
        tmpF.row(f_counter++) = RowVector3i(v_start + 1, v_start + 0, v_start + 2);
        tmpF.row(f_counter++) = RowVector3i(v_start + 1, v_start + 2, v_start + 3);

        // right
        tmpF.row(f_counter++) = RowVector3i(v_start + 4, v_start + 1, v_start + 3);
        tmpF.row(f_counter++) = RowVector3i(v_start + 4, v_start + 3, v_start + 5);

        // bottom 
        tmpF.row(f_counter++) = RowVector3i(v_start + 0, v_start + 1, v_start + 6);                    
        tmpF.row(f_counter++) = RowVector3i(v_start + 1, v_start + 4, v_start + 6);

        // left
        tmpF.row(f_counter++) = RowVector3i(v_start + 0, v_start + 6, v_start + 2);
        tmpF.row(f_counter++) = RowVector3i(v_start + 6, v_start + 7, v_start + 2);

        // top
        tmpF.row(f_counter++) = RowVector3i(v_start + 7, v_start + 5, v_start + 2);
        tmpF.row(f_counter++) = RowVector3i(v_start + 5, v_start + 3, v_start + 2);

        // back
        tmpF.row(f_counter++) = RowVector3i(v_start + 6, v_start + 5, v_start + 7);
        tmpF.row(f_counter++) = RowVector3i(v_start + 6, v_start + 4, v_start + 5);
    }


    void triangulate(Eigen::MatrixXd &new_V, Eigen::MatrixXi &new_F) {
        MatrixXd tmpV;
        MatrixXi tmpF;
        tmpV.resize(resolution*resolution*resolution*8, 3);
        tmpF.resize(resolution*resolution*resolution*12, 3);
        int v_counter = 0, f_counter = 0;
        std::cout << "triangulation progress: " << endl;

        int filled_boxes = 0;
        for (int x = 0; x < resolution; x++) {
            for (int y = 0; y < resolution; y++) {
                for (int z = 0; z < resolution; z++) {
                    if (grid[x][y][z].sdf >= 0 || !grid[x][y][z].filled) continue; 
                    filled_boxes++;

                    int v_start = v_counter;
                    RowVector3d v0(m(0) + x*dx, m(1) + y*dy, m(2) + z*dz);
                    addBoxPoints(tmpV, v_counter, v0);
                    addFaces(tmpF, f_counter, v_start);
                }
            }
        }
        std::cout << "triangulation process finished with : " << filled_boxes << " filled boxes" << std::endl;


        new_V = tmpV.block(0,0, v_counter, 3);
        new_F = tmpF.block(0,0, f_counter, 3);
    }

    void optimize() {

    }
    

};



// END VOXALIZATION

