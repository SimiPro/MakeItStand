#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <vector>
#include <deque>
#include <limits>

#include <igl/readOBJ.h>
#include <igl/writeOFF.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/per_face_normals.h>
#include <igl/centroid.h>
#include <igl/copyleft/cgal/intersect_with_half_space.h>
#include <igl/signed_distance.h>

#include "voxel3.h"
#include "mass_props.h"
#include "optim.h"

bool DEBUG = true;

using namespace std;
using namespace Eigen;


typedef igl::opengl::glfw::Viewer Viewer;




//before rotation and cut
Eigen::MatrixXd V_original;
//before cut
Eigen::MatrixXd V_base;
Eigen::MatrixXd FN_base;
Eigen::MatrixXi F_base;

Eigen::MatrixXd V;
Eigen::MatrixXd N;
Eigen::MatrixXi F;
Eigen::MatrixXd FN;

// Voxalization
MatrixXd new_V; 
MatrixXi new_F;
MatrixXd new_N;

Eigen::MatrixXd planeV;
Eigen::MatrixXd planeFN;
Eigen::MatrixXi planeF;

Eigen::RowVector3d com;
Eigen::RowVector3d gravity;
float gra_xy, gra_xz;
bool gravity_is_set = false;

//Eigen::Vector3d gravity;
//Eigen::Vector3d gravity_from; 
//Eigen::Vector3d gravity_to;

int resolution = 20;
int voxel_max_depth = 3;
double eps_to_boundary = 0.1;

bool set_gravity = false;

bool cleared = true;

bool has_plane = false;
bool set_balance_spot = false;

float y_move_balance_spot = 0.;
float rotate_balancing_spot = 0.;

// bounding box needed for some displaying stuff
Eigen::MatrixXd V_box;

double mouse_down_x;
double mouse_down_y;

void update_com() {
    Eigen::VectorXd s10;
    props(V, F, 0.1,  s10);
    com = getCoM(s10).transpose();
}

void closest_point(const Eigen::RowVector3d p, Eigen::RowVector3d &np) {
    Eigen::MatrixXd pts_neg = V.rowwise() - p;
    Eigen::VectorXd nn = pts_neg.rowwise().squaredNorm();
    Eigen::MatrixXd::Index index;
    nn.minCoeff(&index);
    np = V.row(index);
}

bool pre_draw(Viewer& viewer) {
    if (cleared) {
        cleared = false;

        viewer.data().add_points(Eigen::RowVector3d(0,0,0), Eigen::RowVector3d(0,1,0));

        if (has_plane) {    
            viewer.selected_data_index = 0;
            viewer.data().set_mesh(planeV, planeF);
            viewer.data().set_normals(planeFN);
        }

        // set mesh
        viewer.selected_data_index = 1;
        viewer.data().set_mesh(V, F);
        viewer.core.align_camera_center(V,F);

        //viewer.data().compute_normals();
        viewer.data().set_face_based(true);

        // 
        update_com();
        viewer.data().add_points(com, Eigen::RowVector3d(0, 0, 1));


    	if (gravity_is_set) {
                Eigen::RowVector3d temp = com + 100 * gravity;
                viewer.data().add_points(temp, Eigen::RowVector3d(1,0,0));
                viewer.data().add_edges(com, temp,Eigen::RowVector3d(1,0,0));
    	}


    }

    return false;
}

void clear(Viewer &viewer) {
    viewer.selected_data_index = 0;
    viewer.data().clear();
    viewer.selected_data_index = 1;
    viewer.data().clear();
    cleared = true;
}

//void update_viewer(Viewer& viewer){
//    viewer.data().clear();
//    viewer.data().set_mesh(V, F);
//    viewer.data().set_face_based(true);
//    if(com_is_set) {
//        viewer.data().add_points(com, Eigen::RowVector3d(0, 0, 1));
//    }
//    if(gravity_is_set) {
//        Eigen::RowVector3d temp = com + 100 * gravity;
//        viewer.data().add_points(temp, Eigen::RowVector3d(1,0,0));
//        viewer.data().add_edges(com, temp,Eigen::RowVector3d(1,0,0));
//    }
//    //if(balance_point_is_set) {
//    //    viewer.data().add_points(balance_point, Eigen::RowVector3d(0,1,0));
//    //}
//}

void set_plane() {
    // Find the bounding box
    Eigen::Vector3d m = V_base.colwise().minCoeff();
    Eigen::Vector3d M = V_base.colwise().maxCoeff();

    planeV.resize(4, 3);
    planeF.resize(2, 3);
    planeFN.resize(2, 3);

    planeV.row(0) = Eigen::RowVector3d(m[0], 0, m[2]);
    planeV.row(1) = Eigen::RowVector3d(M[0], 0, m[2]);
    planeV.row(2) = Eigen::RowVector3d(m[0], 0, M[2]);
    planeV.row(3) = Eigen::RowVector3d(M[0], 0, M[2]);

    planeF.row(0) = Eigen::RowVector3i(0, 1, 2);
    planeF.row(1) = Eigen::RowVector3i(1, 3, 2);

    planeFN.row(0) = Eigen::RowVector3d(0,1,0);
    planeFN.row(1) = Eigen::RowVector3d(0,1,0);

    has_plane = true;
}

bool mouse_down(Viewer& viewer, int button, int modifier) {
    mouse_down_x = viewer.current_mouse_x;
    mouse_down_y = viewer.core.viewport(3) - viewer.current_mouse_y;



    if (set_balance_spot) {
        Eigen::Vector3f baryC;
        int fid;
        // Cast a ray in the view direction starting from the mouse position
        double x = viewer.current_mouse_x;
        double y = viewer.core.viewport(3) - viewer.current_mouse_y;
        if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view,
            viewer.core.proj, viewer.core.viewport, V, F, fid, baryC)) {

            // max bary coords, get nearearst vertex
            long c; baryC.maxCoeff(&c);
            Eigen::RowVector3d nn_c = V.row(F(fid,c));
            V.rowwise() -= nn_c;
            V_base.rowwise() -= nn_c;
            clear(viewer);
            set_balance_spot = false;
            set_plane();
        }
    }

    if (set_gravity) return true;

    return false;
}

bool mouse_up(Viewer& viewer, int button, int modifier) {
    double x = viewer.current_mouse_x;
    double y = viewer.core.viewport(3) - viewer.current_mouse_y;

    if (DEBUG) {
        cout << "mouse up x: " << x << endl;
        cout << "mouse up y: " << y << endl;
    }

    //if (set_gravity) {
    //    gravity_from = {mouse_down_x, mouse_down_y, 0};
    //    gravity_to = {x, y, 0};
    //    gravity = (gravity_to - gravity_from).normalized();
    //    Eigen::MatrixXd tmp; tmp.resize(2,3);
    //    tmp.row(0) = V_box.row(0);
    //    double tmp_length = (V_box.row(0) - V_box.row(1)).norm() / 5;
    //    tmp.row(1) = V_box.row(0) + gravity.transpose() * tmp_length;
    //    viewer.data().add_points(tmp, Eigen::RowVector3d(1,0,0));
    //    viewer.data().add_edges(tmp.row(0), tmp.row(1),Eigen::RowVector3d(1,0,0));
    //    set_gravity = false;
    //    return true;
    //}

    return false;
}


bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        double vol;

    //std::cout << "Start: compute center of mass" << std::endl;
        igl::centroid(V, F, com, vol); 

        cout << "igl com: " << endl;
        cout << com << endl;

        Eigen::VectorXd s10;
        props(V, F, 0.1,  s10);


        Vector3d com = getCoM(s10);
        cout << "our com: " << endl;
        cout << com << endl;

    //std::cout << "Done" << std::endl;

        clear(viewer);
    } else if (key == '2') {
        Vector3d p(0,0,0);
        Vector3d n(0,-1,0);
        MatrixXd VC;
        MatrixXi FC;
        MatrixXi J; 
        igl::copyleft::cgal::intersect_with_half_space(V, F_base, p,  n, VC, FC, J);

        clear(viewer);
        V = VC;
        F = FC;
        //viewer.data().set_mesh(VC, FC);
    } else if (key == '3') {
        Voxalization voxal(V, F, resolution, voxel_max_depth, eps_to_boundary);
        // just display voxelization
        voxal.triangulate(new_V, new_F);
        igl::per_face_normals(new_V, new_F, new_N);
        clear(viewer);
        cleared =  false;
        viewer.data().set_mesh(new_V, new_F);
        viewer.data().set_normals(new_N);

        viewer.append_mesh();
        viewer.data().set_mesh(V, F);

        VectorXd s10; 
        props(new_V, new_F, 0.1,  s10);
        cout << "s10:" << endl;
        cout << s10 << endl;    
        cout << "com1: " << endl;
        cout << getCoM(s10) << endl;



        // now calculate center of mass by adding all together
        //vector<MatrixXd> faceNormals;
        //vector<MatrixXi> boxes;
        //voxal.triangulate_with_vectors(new_V, faceNormals, boxes);


//        Eigen::VectorXd s10_all; s10_all.resize(10); s10_all.setZero();

  //      for (int i = 0; i < boxes.size(); i++) {
    //        VectorXd s10; 
     //       props(new_V, boxes[i], faceNormals[i], 0.1,  s10);    
      //      s10_all += s10;
       // }
        

        RowVector3d vCom = getCoM(s10).transpose();
        viewer.data().add_points(vCom, Eigen::RowVector3d(0, 0, 1));
        cout << "com2: " << endl;
        cout << vCom << endl;

    } else if (key == '4') { // optimize! :)
        // 1. calculate mass properties over whole mesh
        // 2. get boundary mesh and get interior mesh (voxalized)
        // 3. optimize interior mesh to align CoM with balancing spot (which should be at 0,0,0)

        // mass properties over whole mesh
        VectorXd s10all;
        props(V, F, 0.1, s10all);

        // voxalization of mesh
        Voxalization voxal(V, F, resolution, voxel_max_depth, eps_to_boundary);

        // get interior mesh
        vector<MatrixXi> boxes; vector<VectorXd> b_s10;
        voxal.get_interior_mesh(new_V, boxes);

        // calculate mass proprties of each box of the interior mesh
        for (int i = 0; i < boxes.size(); i++) {
            VectorXd s10; 
            props(new_V, boxes[i], 0.1,  s10);    
            b_s10.push_back(s10);
        }

        // optimize boxes of interior mesh
        VectorXd betas;
        optim(s10all, b_s10, betas);

        // empty those boxes
        double EPS = 1e-3;
        for (int i = 0; i < betas.rows(); i++) {
            if (betas[i] > 1 - EPS) {
                voxal.empty_box(i);
            }
        }

        voxal.triangulate(new_V, new_F);

        // triangulate mesh
        MatrixXd emptyV;
        MatrixXi emptyF;
        voxal.triangulate_empty(emptyV, emptyF);
        igl::writeOFF("empty_mesh.off", emptyV, emptyF);
        clear(viewer);
        cleared =  false;
        viewer.data().set_mesh(new_V, new_F);

        viewer.append_mesh();
        viewer.data().set_mesh(V, F);

        voxal.writeCAD(V, F);

        VectorXd s10; 
        props(emptyV, emptyF, 0.1,  s10);
        RowVector3d vCom = getCoM(s10all - s10).transpose();
        viewer.data().add_points(vCom, Eigen::RowVector3d(0, 0, 1));
        cout << "com: " << endl;
        cout << vCom << endl;
    } else if (key == '9') {
        igl::writeOFF("saved_file.off", V, F);
    } 
}



void setBoudingBox() {
    // Find the bounding box
    Eigen::Vector3d m = V.colwise().minCoeff();
    Eigen::Vector3d M = V.colwise().maxCoeff();

    // Corners of the bounding box
    V_box <<
    m(0), m(1), m(2),
    M(0), m(1), m(2),
    M(0), M(1), m(2),
    m(0), M(1), m(2),
    m(0), m(1), M(2),
    M(0), m(1), M(2),
    M(0), M(1), M(2),
    m(0), M(1), M(2);

    if (DEBUG) {
        cout << "Mesh bounding box: " << endl;
        cout << V_box << endl;
        cout << "End Mesh bounding box" << endl;
    }

    eps_to_boundary = abs((M - m).maxCoeff())*0.01;
}

void update_gravity() {
    double x, y, z;
    x = std::cos(gra_xz / 180 * M_PI) * std::sin(gra_xy / 180 * M_PI);
    y = -std::cos(gra_xy / 180. * M_PI);
    z = std::sin(gra_xz / 180 * M_PI) * std::sin(gra_xy / 180. * M_PI);
    gravity = Eigen::RowVector3d(x, y, z);
    gravity_is_set = true;
}

void align_gravity() {
    //rotate model such that gravity vector and neg. y-axis are aligned
    //translate com to origin (not necessary, but good for better visibility)

    //rotation
    Eigen::Matrix3d rot_mat;
    Eigen::Vector3d y(0, -1, 0);
    //double angle = std::acos(y.dot(gra));
    //Eigen::Vector3d axis = y.cross(gra);
    //Eigen::AngleAxisd rot(-angle, axis);
    //rot_mat = rot.toRotationMatrix();
    Eigen::Quaterniond quat = Eigen::Quaterniond().setFromTwoVectors(gravity, y);
    rot_mat = quat.toRotationMatrix();
    V = V_base * rot_mat.transpose();
    gravity = gravity * rot_mat.transpose();

    //translation
    update_com();
   // V = V.rowwise() - com;
    //com = Eigen::RowVector3d(0, 0, 0);
    //V_base = V;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage ./bin <mesh.off>" << endl;
        igl::readOFF("../data/sphere.off",V,F,N);
    } else {
        // Read points and normals
        string filename(argv[1]);
        if (filename.find(".obj") == string::npos) {
            igl::readOFF(filename, V, F, N);
        } else {
            MatrixXd TC;
            MatrixXi FTC;
            igl::readOBJ(filename,V, TC, N, F, FTC, FN);    
        }

        
    }

    // SIMPLE BOX
        
    //


    igl::per_face_normals(V,F, FN);
    V_original = V;
    V_base = V;
    FN_base = FN;
    F_base = F;

    V_box.resize(8,3); V_box.setZero();     
    setBoudingBox();


    // init viewer
    Viewer viewer;

    // zero com
    com.resize(3); com.setZero();

    // set gravity to 0 for start
    gravity.resize(3); gravity.setZero();

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_custom_window = [&]() {
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 400), ImGuiSetCond_FirstUseEver);
        ImGui::Begin(
            "Window Handling Stuff", nullptr,
            ImGuiWindowFlags_NoSavedSettings
        );   


        ImGui::Checkbox("Set Balance Spot", &set_balance_spot);
        ImGui::DragFloat("Move spot up/down", &y_move_balance_spot, 0.1);
        if (ImGui::IsItemActive()) {
            V  = V_base.rowwise() - RowVector3d(0, y_move_balance_spot, 0);
            clear(viewer);
            set_plane();
        }

        ImGui::InputInt("Resolution", &resolution);
        ImGui::InputInt("Octree depth", &voxel_max_depth); 
        ImGui::InputDouble("Eps voxel vs boundary", &eps_to_boundary); 



	//gravity
        ImGui::SliderFloat("Gravity angle in xy-plane", &gra_xy, -180.f, 180.f);
        if (ImGui::IsItemActive()) {
            update_gravity();
            clear(viewer);
            set_plane();
        }
        ImGui::SliderFloat("Gravity angle in xz-plane", &gra_xz, -180.f, 180.f);
        if (ImGui::IsItemActive()) {
            update_gravity();
            clear(viewer);
            set_plane();
        }
        if (ImGui::Button("Update gravity", ImVec2(-1,0))){
            update_gravity();
            //update_viewer(viewer);
            clear(viewer);
        }
        if (ImGui::Button("rotate model to align gravity", ImVec2(-1,0))) {
            align_gravity();
            //update_viewer(viewer);
            clear(viewer);
        }


        ImGui::SliderAngle("Rotate around balancing spot", &rotate_balancing_spot);

        //ImGui::Checkbox("Set Gravity", &set_gravity);
        //ImGui::InputDouble("Gravity x", &gravity[0], 0., 0);
        //ImGui::InputDouble("Gravity y", &gravity[1], 0., 0);
        //ImGui::InputDouble("Gravity z", &gravity[2], 0., 0);


        ImGui::End();
    };

    
    viewer.callback_mouse_down = &mouse_down;
    viewer.callback_mouse_up = &mouse_up;
    viewer.callback_key_down = &callback_key_down;
    viewer.callback_pre_draw = &pre_draw;

    // Plot the mesh

    //viewer.data().set_mesh(V, F);
    //viewer.data().set_face_based(true);

    viewer.append_mesh(); // now we have 2 mesh
    // mesh 0 = plane
    // mesh 1 = mesh
    viewer.launch();

    clear(viewer);
}
