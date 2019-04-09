#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <vector>
#include <deque>

#include <igl/readOBJ.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/per_face_normals.h>
#include <igl/centroid.h>
#include <igl/copyleft/cgal/intersect_with_half_space.h>

#include "mass_props.h"

bool DEBUG = true;

using namespace std;

typedef igl::opengl::glfw::Viewer Viewer;


Eigen::MatrixXd V_base;
Eigen::MatrixXd FN_base;
Eigen::MatrixXi F_base;

Eigen::MatrixXd V;
Eigen::MatrixXd N;
Eigen::MatrixXi F;
Eigen::MatrixXd FN;

Eigen::MatrixXd planeV;
Eigen::MatrixXd planeFN;
Eigen::MatrixXi planeF;

Eigen::RowVector3d com;
bool com_is_set = false;

Eigen::Vector3d gravity;
Eigen::Vector3d gravity_from; 
Eigen::Vector3d gravity_to;

int num_handles = 5;
vector<Eigen::Vector3d> handles;
vector<bool> used_handles(num_handles, false);
deque<bool> set_handles(num_handles, false);
vector<Eigen::Vector3d> p_handles;

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

void closest_point(const Eigen::RowVector3d p, Eigen::RowVector3d &np) {
    Eigen::MatrixXd pts_neg = V.rowwise() - p;
    Eigen::VectorXd nn = pts_neg.rowwise().squaredNorm();
    Eigen::MatrixXd::Index index;
    nn.minCoeff(&index);
    np = V.row(index);
}

bool pre_draw(Viewer& viewer) {
    if (cleared) {
        if (has_plane) {    
            viewer.selected_data_index = 0;
            viewer.data().set_mesh(planeV, planeF);
            viewer.data().set_normals(planeFN);
        }

        // set mesh
        viewer.selected_data_index = 1;
        viewer.data().set_mesh(V, F);
        //viewer.data().compute_normals();
        viewer.data().set_face_based(true);

        // 
        viewer.data().add_points(com, Eigen::RowVector3d(0, 0, 1));

        cleared = false;

        for (int i = 0; i < used_handles.size(); i++) {
            if (used_handles[i])
                viewer.data().add_points(p_handles[i], Eigen::RowVector3d(1, 0, 0));
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

void update_viewer(Viewer& viewer){
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    if(com_is_set) {
        viewer.data().add_points(com, Eigen::RowVector3d(0, 0, 1));
    }
    //if(gravity_is_set) {
    //    Eigen::RowVector3d temp = com + 100 * gravity;
    //    viewer.data().add_points(temp, Eigen::RowVector3d(1,0,0));
    //    viewer.data().add_edges(com, temp,Eigen::RowVector3d(1,0,0));
    //}
    //if(balance_point_is_set) {
    //    viewer.data().add_points(balance_point, Eigen::RowVector3d(0,1,0));
    //}
}

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
    if (DEBUG) {
        cout << "mouse down x: " << mouse_down_x << endl;
        cout << "mouse down y: " << mouse_down_y << endl;
    }
    for (int i =0; i < num_handles; i++) {
        if (!set_handles[i]) continue;
        set_handles[i] = false;
        int fid;
        Eigen::Vector3f baryC;
        // Cast a ray in the view direction starting from the mouse position
        double x = viewer.current_mouse_x;
        double y = viewer.core.viewport(3) - viewer.current_mouse_y;
        if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view,
            viewer.core.proj, viewer.core.viewport, V, F, fid, baryC)) {

            // max bary coords, get nearearst vertex
            long c; baryC.maxCoeff(&c);
            Eigen::RowVector3d nn_c = V.row(F(fid,c));
            used_handles[i] = true;
            p_handles[i] = nn_c;
        }
        return true;
    }


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

    if (set_gravity) {
        gravity_from = {mouse_down_x, mouse_down_y, 0};
        gravity_to = {x, y, 0};
        gravity = (gravity_to - gravity_from).normalized();
        Eigen::MatrixXd tmp; tmp.resize(2,3);
        tmp.row(0) = V_box.row(0);
        double tmp_length = (V_box.row(0) - V_box.row(1)).norm() / 5;
        tmp.row(1) = V_box.row(0) + gravity.transpose() * tmp_length;
        viewer.data().add_points(tmp, Eigen::RowVector3d(1,0,0));
        viewer.data().add_edges(tmp.row(0), tmp.row(1),Eigen::RowVector3d(1,0,0));
        set_gravity = false;
        return true;
    }

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
        props(V, F, FN, 0.1,  s10);


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
        has_plane = false;

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
}

void update_com() {
    double vol;
    igl::centroid(V, F, com, vol);
    com_is_set = true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage ./bin <mesh.off>" << endl;
        igl::readOFF("../data/sphere.off",V,F,N);
    } else {
        // Read points and normals
        igl::readOFF(argv[1],V,F,N);
    }

    igl::per_face_normals(V,F, FN);
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

	//center of mass
	if (ImGui::Button("Update center of mass", ImVec2(-1,0))){
            update_com();
	    update_viewer(viewer);
        }

        ImGui::Checkbox("Set Balance Spot", &set_balance_spot);
        ImGui::DragFloat("Move spot up/down", &y_move_balance_spot, 0.1);
        if (ImGui::IsItemActive()) {
            V  = V_base.rowwise() - RowVector3d(0, y_move_balance_spot, 0);
            clear(viewer);
            set_plane();
        }

        ImGui::SliderAngle("Rotate around balancing spot", &rotate_balancing_spot);

        ImGui::Checkbox("Set Gravity", &set_gravity);
        ImGui::InputDouble("Gravity x", &gravity[0], 0., 0);
        ImGui::InputDouble("Gravity y", &gravity[1], 0., 0);
        ImGui::InputDouble("Gravity z", &gravity[2], 0., 0);




        for (int i = 0; i < num_handles; i++) {
            p_handles.push_back({0,0,0});      
        }

        for (int i = 0; i < used_handles.size(); i++) {
            ImGui::Checkbox("Set Handle", &(set_handles[i]));
            ImGui::InputDouble("Z-depth", &p_handles[i][2], 0, 0);
        }


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
}
