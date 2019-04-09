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

bool DEBUG = true;

using namespace std;

typedef igl::opengl::glfw::Viewer Viewer;

Eigen::MatrixXd V;
Eigen::MatrixXd N;
Eigen::MatrixXi F;
Eigen::MatrixXd FN;

Eigen::MatrixXd V_orig;
Eigen::MatrixXd N_orig;
Eigen::MatrixXi F_orig;
Eigen::MatrixXd FN_orig;

Eigen::RowVector3d com;
Eigen::RowVector3d gra = Eigen::RowVector3d(0, -1, 0);
Eigen::RowVector3d balance_point;
float gra_x = 0;
float gra_y = 0;
float cut_heigth;

//Eigen::Vector3d gravity;
//Eigen::Vector3d gravity_from; 
//Eigen::Vector3d gravity_to;

int num_handles = 5;
vector<Eigen::Vector3d> handles;
vector<bool> used_handles(num_handles, false);
deque<bool> set_handles(num_handles, false);
vector<Eigen::Vector3d> p_handles;

bool set_gravity = false;
bool set_balance_point = false;

bool com_is_set = false;
bool gravity_is_set = false;
bool balance_point_is_set = false;

bool cleared = false;

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

void update_viewer(Viewer& viewer){
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    if(com_is_set) {
        viewer.data().add_points(com, Eigen::RowVector3d(0, 0, 1));
    }
    if(gravity_is_set) {
        Eigen::RowVector3d temp = com + 100 * gra;
        viewer.data().add_points(temp, Eigen::RowVector3d(1,0,0));
        viewer.data().add_edges(com, temp,Eigen::RowVector3d(1,0,0));
    }
    if(balance_point_is_set) {
        viewer.data().add_points(balance_point, Eigen::RowVector3d(0,1,0));
    }
}

bool pre_draw(Viewer& viewer) {
    if (cleared) {
        // set mesh
        viewer.data().set_mesh(V, F);
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
    viewer.data().clear();
    cleared = true;
}




bool mouse_down(Viewer& viewer, int button, int modifier) {
    mouse_down_x = viewer.current_mouse_x;
    mouse_down_y = viewer.core.viewport(3) - viewer.current_mouse_y;
    if (DEBUG) {
        cout << "mouse down x: " << mouse_down_x << endl;
        cout << "mouse down y: " << mouse_down_y << endl;
    }
    if (set_balance_point) {
	int fid;
	Eigen::RowVector3f baryC;
        // Cast a ray in the view direction starting from the mouse position
        double x = viewer.current_mouse_x;
        double y = viewer.core.viewport(3) - viewer.current_mouse_y;
        if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view,
            viewer.core.proj, viewer.core.viewport, V, F, fid, baryC)) {

            // max bary coords, get nearearst vertex
            long c; baryC.maxCoeff(&c);
            Eigen::RowVector3d nn_c = V.row(F(fid,c));
            balance_point = nn_c;
	    balance_point_is_set = true;
	    update_viewer(viewer);
	    //viewer.data().add_points(balance, Eigen::RowVector3d(0,1,0));
        }

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
	update_viewer(viewer);
        return true;
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

//bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
//    if (key == '1') {
//        double vol;
//
//    //std::cout << "Start: compute center of mass" << std::endl;
//        igl::centroid(V, F, com, vol); 
//    //std::cout << "Done" << std::endl;
//
//        clear(viewer);
//    }
//}



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
    //std::cout << "Start: compute center of mass" << std::endl;
    igl::centroid(V, F, com, vol);
    com_is_set = true;
    //std::cout << "Done" << std::endl;
}

void update_gravity() {
    double x, y, z;
    x = std::cos(gra_y / 180 * M_PI) * std::sin(gra_x / 180 * M_PI);
    y = -std::cos(gra_x / 180. * M_PI);
    z = std::sin(gra_y / 180 * M_PI) * std::sin(gra_x / 180. * M_PI);
    gra = Eigen::RowVector3d(x, y, z);
    gravity_is_set = true;
}

void update_balance_surface() {

    V = V_orig;
    update_com();

    //rotation
    Eigen::MatrixXd rot_mat;
    Eigen::Vector3d y(0, -1, 0);
    double angle = std::acos(y.dot(gra));
    Eigen::Vector3d axis = y.cross(gra);
    std::cout << "axis = " << axis << std::endl;
    std::cout << "angle = " << angle / M_PI * 180. << std::endl;
    //Eigen::AngleAxisd rot(-angle, axis);
    Eigen::Quaterniond quat = Eigen::Quaterniond().setFromTwoVectors(gra, y);
    //rot_mat = rot.toRotationMatrix();
    rot_mat = quat.toRotationMatrix();
    //rotate 
    V = V * rot_mat.transpose();
    //com = com * rot_mat.transpose();
    gra = gra * rot_mat.transpose();
    balance_point = balance_point * rot_mat.transpose();
    std::cout << "gravity = " << gra  << std::endl;

    //translation
    Eigen::RowVector3d cut(0, cut_heigth, 0);
    std::cout << "translate V" << std::endl;
    V = V.rowwise() - (balance_point + cut);
    std::cout << "translate balance_point" << std::endl;
    balance_point = Eigen::RowVector3d(0, 0, 0);
    std::cout << "gravity = " << gra  << std::endl;

    //cut
    std::cout << "Start: cut" << std::endl;
    for (int i = 0; i < V.rows(); i++) {
        if (V.row(i)[1] < 0) {
		V.row(i)[1] = 0.;
	}
    }
    std::cout << "Done" << std::endl;

    update_com();
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage ./bin <mesh.off>" << endl;
        igl::readOFF("../data/sphere.off",V,F,N);
	V_orig = V;
	F_orig = F;
	N_orig = N;
    } else {
        // Read points and normals
        igl::readOFF(argv[1],V,F,N);
        V_orig = V;
	F_orig = F;
	N_orig = N;
    }

    igl::per_face_normals(V,F, FN);
    FN_orig = FN;

    V_box.resize(8,3); V_box.setZero();     
    setBoudingBox();

    // init viewer
    Viewer viewer;

    // zero com
    com.resize(3); com.setZero();


    // set gravity to 0 for start
    //gravity.resize(3); gravity.setZero();

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_custom_window = [&]() {
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(250, 600), ImGuiSetCond_FirstUseEver);
        ImGui::Begin(
            "Window Handling Stuff", nullptr,
            ImGuiWindowFlags_NoSavedSettings
        );   

        //ImGui::Checkbox("Set Gravity", &set_gravity);
        //ImGui::InputDouble("Gravity x", &gravity[0], 0., 0);
        //ImGui::InputDouble("Gravity y", &gravity[1], 0., 0);
        //ImGui::InputDouble("Gravity z", &gravity[2], 0., 0);
        if (ImGui::Button("Update center of gravity", ImVec2(-1,0)))
        {
            update_com();
	    update_viewer(viewer);
        }
	ImGui::SliderFloat("Gravity angle", &gra_x, -180.f, 180.f);
	ImGui::SliderFloat("Gravity rotation", &gra_y, -180.f, 180.f);
	// Add a button
        if (ImGui::Button("Update gravity", ImVec2(-1,0)))
        {
            update_gravity();
            update_viewer(viewer);
        }

	ImGui::Checkbox("Set balance point", &set_balance_point);
        ImGui::SliderFloat("balance surface cut off", &cut_heigth, 0.f, 1.f);
	if (ImGui::Button("Update balance surface", ImVec2(-1,0)))
        {
            update_balance_surface();
            update_viewer(viewer);
        }

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
    //viewer.callback_key_down = &callback_key_down;
    viewer.callback_pre_draw = &pre_draw;

    // Plot the mesh

    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    viewer.launch();
}
