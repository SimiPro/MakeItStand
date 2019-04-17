#include <vector>

using namespace std;
using namespace Eigen;


// VOXALIZATION 

struct Tripletz {
    int x,y,z;
};

struct Box {
    bool filled;
    double sdf;
    bool is_boundary;
    double dx, dy, dz;
    int depth; 
    Vector3d center;
    vector<Box*> children; 

    Box(bool filled_, double sdf_, bool is_boundary_, double dx_, double dy_, double dz_): 
        filled(filled_), sdf(sdf_), is_boundary(is_boundary_), dx(dx_), dy(dy_), dz(dz_), depth(0) {}

    Box(Box* parent, int d1, int d2, int d3) {
        assert((d1 == -1 || d1 == 1) && (d2 == -1 || d2 == 1) || (d3 == -1 || d3 == 1));
        filled = true;
        sdf = 0;
        is_boundary = false;
        depth = parent->depth + 1;

        // set child box on half of the parent and make center depending on d1, d2 ,d3
        dx = parent->dx/2, dy = parent->dy/2, dz = parent->dz/2;
        center = parent->center + Vector3d(d1*dx/2, d2*dy/2, d3*dz/2);
        
    }
    
};


typedef vector<vector<vector<Box > > > Grid;
typedef vector<vector<Box>> VII;
typedef vector<Box> VI;

class Voxalization {
    const MatrixXd &V;
    MatrixXi &F;
    RowVector3d &com;
    int resolution;
    Vector3d m;
    Vector3d M;
    Grid grid;
    double dx, dy, dz;
    vector<Tripletz> box_id_to_grid_id; 
    int max_depth;
    int num_boxes;
    vector<Box*> boxes;


public:
    Voxalization(MatrixXd &V_, MatrixXi &F_, int resolution_, RowVector3d &com_, int max_depth_): V(V_), F(F_), resolution(resolution_),
                com(com_), max_depth(max_depth_), num_boxes(0) {

        // BOUNDING BOX
        m = V.colwise().minCoeff();
        M = V.colwise().maxCoeff();

        dx = (M(0) - m(0)) / resolution;
        dy = (M(1) - m(1)) / resolution;
        dz = (M(2) - m(2)) / resolution;

        grid = Grid(resolution_, VII(resolution_, VI(resolution_, {true, 0, false, dx, dy, dz})));
        
        // BUILD QUERY POINTS
        vector<Box*> unfinished;

        for (int x = 0; x < resolution; x++) {
            for (int y = 0; y < resolution; y++) {
                for (int z = 0; z < resolution; z++) {
                    Vector3d box_center(m(0) + (x + 0.5)*dx, m(1) + (y + 0.5)*dy, m(2) + (z + 0.5)*dz);            
                    grid[x][y][z].center = box_center;
                    unfinished.push_back(&(grid[x][y][z]));
                    num_boxes++;
                }
            }
        }
        // 
        build_child_boxes(unfinished);

        cout << "Built voxalization with: " << boxes.size() << " boxes" << endl;

    }

    void build_child_boxes(const vector<Box*> &unfinished) {
         // Ask that query
        
        Eigen::MatrixXd Q;
        Q.resize(unfinished.size(), 3);
        for (int i = 0; i < unfinished.size(); i++) {
            Box *box = unfinished[i];
            Q.row(i) = box->center;
        }

        VectorXd S, I;
        MatrixXd C, N;

        igl::SignedDistanceType type = igl::SignedDistanceType::SIGNED_DISTANCE_TYPE_PSEUDONORMAL;
        igl::signed_distance(Q, V, F, type,
            numeric_limits<float>::min(), numeric_limits<float>::max(), S, I, C, N);


        // set in/out
        vector<Box*> new_unfinished;
        for (int i = 0; i < unfinished.size(); i++) {
            Box* box = unfinished[i];
            box->sdf = S[i];
            box->is_boundary = pow(box->sdf,2) <= 2*pow(box->dx/2, 2) + pow(box->dy/2, 2) + pow(box->dz/2, 2);
            if (box->is_boundary && box->depth < max_depth) {
                // remove the "parent" and replace it with 8 children
                Box *child0 = new Box(box, -1,-1,-1);
                Box *child1 = new Box(box, -1,-1,1);
                Box *child2 = new Box(box, -1,1,-1);
                Box *child3 = new Box(box, -1,1,1);
                Box *child4 = new Box(box, 1,-1,-1);
                Box *child5 = new Box(box, 1,-1,1);
                Box *child6 = new Box(box, 1,1,-1);
                Box *child7 = new Box(box, 1,1,1);

                box->children.push_back(child0);
                box->children.push_back(child1);
                box->children.push_back(child2);
                box->children.push_back(child3);
                box->children.push_back(child4);
                box->children.push_back(child5);
                box->children.push_back(child6);
                box->children.push_back(child7);

                new_unfinished.push_back(child0);
                new_unfinished.push_back(child1);
                new_unfinished.push_back(child2);
                new_unfinished.push_back(child3);
                new_unfinished.push_back(child4);
                new_unfinished.push_back(child5);
                new_unfinished.push_back(child6);
                new_unfinished.push_back(child7);
            } else {
                boxes.push_back(box);
            }
        }

        if (new_unfinished.size() > 0)
            build_child_boxes(new_unfinished);
    }

    void empty_box(int id) {
        assert(id < box_id_to_grid_id.size() && id >= 0);
        Tripletz trip = box_id_to_grid_id[id];
        grid[trip.x][trip.y][trip.z].filled = false;

    }


    void get_interior_mesh(Eigen::MatrixXd &new_V, vector<MatrixXi> &faces) {
        MatrixXd tmpV;
        tmpV.resize(num_boxes*8, 3);

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
                   // addBoxPoints(tmpV, v_counter, v0);
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

    void addBoxPoints(MatrixXd &tmpV, int &v_counter, const RowVector3d &v0, const Box* box) {
        tmpV.row(v_counter++) = v0; // 0
        tmpV.row(v_counter++) = v0 + RowVector3d(box->dx, 0, 0); // 1
        tmpV.row(v_counter++) = v0 + RowVector3d(0, box->dy, 0); // 2
        tmpV.row(v_counter++) = v0 + RowVector3d(box->dx, box->dy, 0); // 3
        tmpV.row(v_counter++) = v0 + RowVector3d(box->dx, 0, box->dz); // 4
        tmpV.row(v_counter++) = v0 + RowVector3d(box->dx, box->dy, box->dz); // 5
        tmpV.row(v_counter++) = v0 + RowVector3d(0, 0, box->dz); // 6
        tmpV.row(v_counter++) = v0 + RowVector3d(0, box->dy, box->dz); // 7
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
        tmpV.resize(boxes.size()*8, 3);
        tmpF.resize(boxes.size()*12, 3);
        int v_counter = 0, f_counter = 0;
        std::cout << "triangulation progress: " << endl;

        int filled_boxes = 0;
        for (int i = 0; i < boxes.size(); i++) {
            Box *box = boxes[i];
            if (box->sdf > 0 || !box->filled) continue; 
            filled_boxes++;
            int v_start = v_counter;
            RowVector3d bottomLeft = box->center - Vector3d(box->dx/2, box->dy/2, box->dz/2);
            addBoxPoints(tmpV, v_counter, bottomLeft,  box);
            addFaces(tmpF, f_counter, v_start);
        }

        /*for (int x = 0; x < resolution; x++) {
            for (int y = 0; y < resolution; y++) {
                for (int z = 0; z < resolution; z++) {
                    if (grid[x][y][z].sdf >= 0 || !grid[x][y][z].filled) continue; 
                    filled_boxes++;

                    int v_start = v_counter;
                    RowVector3d v0(m(0) + x*dx, m(1) + y*dy, m(2) + z*dz);
                    
                }
            }
        }*/
        std::cout << "triangulation process finished with : " << filled_boxes << " filled boxes" << std::endl;


        new_V = tmpV.block(0,0, v_counter, 3);
        new_F = tmpF.block(0,0, f_counter, 3);
    }

    void optimize() {

    }
    

};



// END VOXALIZATION
