#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <vector>

#include <igl/readOBJ.h>
#include <igl/unproject_onto_mesh.h>

#include <igl/centroid.h>

bool DEBUG = true;

using namespace std;

typedef igl::opengl::glfw::Viewer Viewer;

Eigen::MatrixXd V, C;
Eigen::MatrixXd N;
Eigen::MatrixXi F;



Eigen::Vector3d gravity;
Eigen::Vector3d gravity_from; 
Eigen::Vector3d gravity_to;

vector<Eigen::Vector3d> handles;

bool set_gravity = false;
bool set_handles = false;

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

bool mouse_down(Viewer& viewer, int button, int modifier) {
    mouse_down_x = viewer.current_mouse_x;
    mouse_down_y = viewer.core.viewport(3) - viewer.current_mouse_y;
    if (DEBUG) {
        cout << "mouse down x: " << mouse_down_x << endl;
        cout << "mouse down y: " << mouse_down_y << endl;
    }
    if (set_handles) {
        int fid;
        Eigen::Vector3f baryC;
        // Cast a ray in the view direction starting from the mouse position
        double x = viewer.current_mouse_x;
        double y = viewer.core.viewport(3) - viewer.current_mouse_y;
        if(igl::unproject_onto_mesh(Eigen::Vector2f(x,y), viewer.core.view,
            viewer.core.proj, viewer.core.viewport, V, F, fid, baryC)) {

            // max bary coords, get nearearst vertex
            //long c; baryC.maxCoeff(&c);
            //Eigen::RowVector3d new_c = V.row(F(fid,c));


            // add some offset in direction of non normal so that we get nearest point on other side
            //Eigen::RowVector3d fNormal = N.row(fid);
            


            // paint hit red
            C.row(fid) << 1, 0, 0;
            viewer.data().set_colors(C);
        }
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
        Eigen::RowVector3d cen;
        double vol;

    //std::cout << "Start: compute center of mass" << std::endl;
        igl::centroid(V, F, cen, vol); 
    //std::cout << "Done" << std::endl;

        viewer.data().add_points(cen, Eigen::RowVector3d(1,0,0));
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

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage ./bin <mesh.off>" << endl;
        igl::readOFF("../data/sphere.off",V,F,N);
    } else {
        // Read points and normals
        igl::readOFF(argv[1],V,F,N);
    }
    C = Eigen::MatrixXd::Constant(F.rows(),3,1);

    V_box.resize(8,3); V_box.setZero();     
    setBoudingBox();

    // init viewer
    Viewer viewer;

    // set gravity to 0 for start
    gravity.resize(3); gravity.setZero();

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_custom_window = [&]() {
        // Define next window position + size
        ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 160), ImGuiSetCond_FirstUseEver);
        ImGui::Begin(
            "Window Handling Stuff", nullptr,
            ImGuiWindowFlags_NoSavedSettings
        );   

        ImGui::Checkbox("Set Gravity", &set_gravity);
        ImGui::InputDouble("Gravity x", &gravity[0], 0., 0);
        ImGui::InputDouble("Gravity y", &gravity[1], 0., 0);
        ImGui::InputDouble("Gravity z", &gravity[2], 0., 0);

        ImGui::Checkbox("Set Handles", &set_handles);


        ImGui::End();
    };

    
    viewer.callback_mouse_down = &mouse_down;
    viewer.callback_mouse_up = &mouse_up;
    viewer.callback_key_down = callback_key_down;

    // Plot the mesh

    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    viewer.launch();
}
