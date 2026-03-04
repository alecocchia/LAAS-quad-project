#include <gz/sim/System.hh>
#include <gz/sim/Entity.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/plugin/Register.hh>
#include <gz/math/Pose3.hh>

// Component Headers
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/Name.hh> 
#include <gz/sim/components/ChildLinkName.hh>
#include <gz/sim/components/Inertial.hh>
#include <gz/sim/components/Gravity.hh>

// Ignition Transport Headers
#include <gz/transport/Node.hh>
#include <gz/msgs/wrench.pb.h>
#include <gz/msgs/actuators.pb.h> 

// ROS 2 Headers
#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <cmath>
#include <Eigen/Dense>
#include <sstream>

namespace mrsim_gazebo_sim
{

    struct Rotor
    {
        gz::sim::Entity joint;
        gz::sim::Entity child_link;
        int spin_direction;
    };
    
    class MrsimDroneIgnPlugin : public gz::sim::System,
                                public gz::sim::ISystemConfigure,
                                public gz::sim::ISystemPreUpdate
    {
    public:
        MrsimDroneIgnPlugin() : rclcpp_node_(nullptr), latest_wrench_command_(), physics_initialized_(false) {}

        void Configure(const gz::sim::Entity &entity,
                       const std::shared_ptr<const sdf::Element> &sdf,
                       gz::sim::EntityComponentManager &ecm,
                       gz::sim::EventManager & /*eventMgr*/) override
        {
            this->model_ = gz::sim::Model(entity);
            if (!this->model_.Valid(ecm))
            {
                RCLCPP_ERROR(rclcpp::get_logger("MrsimDroneIgnPlugin"), "MrsimDroneIgnPlugin dovrebbe essere attaccato a un'entità modello.");
                return;
            }

            if (!rclcpp::ok())
            {
                rclcpp::init(0, nullptr);
            }
            this->rclcpp_node_ = rclcpp::Node::make_shared(
                this->model_.Name(ecm) + "_plugin_node");

            RCLCPP_INFO(this->rclcpp_node_->get_logger(), "MrsimDroneIgnPlugin caricato per il modello [%s]", this->model_.Name(ecm).c_str());
            
            std::string wrench_topic_name = sdf->Get<std::string>("wrench_topic", "/model/qr4/wrench_cmd").first;
            std::string motor_speed_topic_name = sdf->Get<std::string>("motor_speed_topic", "/gazebo/command/motor_speed").first;

            this->motor_speed_pub_ = this->gz_node_.Advertise<gz::msgs::Actuators>("/qr4/gazebo/command/motor_speed");
            if (!this->motor_speed_pub_)
            {
                RCLCPP_ERROR(this->rclcpp_node_->get_logger(), "Impossibile creare il publisher per il topic delle velocità dei motori [%s]", motor_speed_topic_name.c_str());
            }

            if (!this->gz_node_.Subscribe(
                wrench_topic_name,
                &MrsimDroneIgnPlugin::OnWrenchCommandsGz, this))
            {
                RCLCPP_ERROR(this->rclcpp_node_->get_logger(), "Errore nella sottoscrizione al topic dei comandi wrench [%s]", wrench_topic_name.c_str());
            }
            
            sdf::ElementPtr rotors_elem = std::const_pointer_cast<sdf::Element>(sdf)->FindElement("rotors");
            if (rotors_elem)
            {
                this->cf_ = rotors_elem->Get<double>("cf", 6.5e-04).first;
                this->ct_ = rotors_elem->Get<double>("ct", 1e-05).first;
                this->L_ = 0.23; 

                for (sdf::ElementPtr rotor_elem = rotors_elem->FindElement("rotor"); rotor_elem; rotor_elem = rotor_elem->GetNextElement("rotor"))
                {
                    sdf::ElementPtr joint_elem = rotor_elem->FindElement("joint");
                    if (joint_elem)
                    {
                        std::string joint_name = joint_elem->Get<std::string>();
                        std::string spin_str = joint_elem->Get<std::string>("spin", "ccw").first;

                        gz::sim::Entity joint_entity = this->model_.JointByName(ecm, joint_name);
                        if (joint_entity != gz::sim::kNullEntity)
                        {
                            Rotor rotor;
                            rotor.joint = joint_entity;
                            rotor.spin_direction = (spin_str == "cw") ? -1 : 1;

                            std::string child_link_name_from_ecm = "";
                            auto childLinkNameComp = ecm.Component<gz::sim::components::ChildLinkName>(rotor.joint);
                            if (childLinkNameComp)
                            {
                                child_link_name_from_ecm = childLinkNameComp->Data();
                            }
                            else
                            {
                                RCLCPP_ERROR(this->rclcpp_node_->get_logger(), "Giunto [%s] non ha un componente ChildLinkName.", joint_name.c_str());
                                continue;
                            }

                            rotor.child_link = this->model_.LinkByName(ecm, child_link_name_from_ecm);
                            if (rotor.child_link != gz::sim::kNullEntity)
                            {
                                this->rotors_.push_back(rotor);
                            }
                        }
                    }
                }
            }

            Eigen::Matrix4d M_alloc;
            M_alloc <<  this->cf_, this->cf_, this ->cf_, this->cf_,
                        0.0,  this->L_*this->cf_,  0.0,  -this->L_*this->cf_,
                        -this->L_*this->cf_,  0.0,  this->L_*this->cf_,  0.0,
                        this->ct_, -this->ct_, this->ct_, -this->ct_;

            try {
                this->M_alloc_inv_ = M_alloc.inverse();
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->rclcpp_node_->get_logger(), "Errore calcolo inversa: %s", e.what());
            }

            // Inizializza hovering a zero (verrà calcolato dinamicamente nel PreUpdate)
            this->hover_wrench_.mutable_force()->set_x(0.0);
            this->hover_wrench_.mutable_force()->set_y(0.0);
            this->hover_wrench_.mutable_force()->set_z(0.0);
            this->hover_wrench_.mutable_torque()->set_x(0.0);
            this->hover_wrench_.mutable_torque()->set_y(0.0);
            this->hover_wrench_.mutable_torque()->set_z(0.0);
            
            // Stato iniziale: a terra e fermo
            this->latest_wrench_command_ = this->hover_wrench_;
            this->last_cmd_time_ = std::chrono::steady_clock::now();
        }

        void PreUpdate(const gz::sim::UpdateInfo &info,
                       gz::sim::EntityComponentManager &ecm) override
        {
            if (info.paused)
                return;

            // --- INIZIALIZZAZIONE FISICA DINAMICA ---
            if (!this->physics_initialized_) {
                double total_mass = 0.0;
                double gravity_z = 9.81; // Fallback

                auto worldEntity = ecm.EntityByComponents(gz::sim::components::Gravity());
                if (worldEntity != gz::sim::kNullEntity) {
                    auto gravityComp = ecm.Component<gz::sim::components::Gravity>(worldEntity);
                    if (gravityComp) {
                        gravity_z = std::abs(gravityComp->Data().Z());
                    }
                }

                std::vector<gz::sim::Entity> links = this->model_.Links(ecm);
                for (const auto &link : links) {
                    auto inertialComp = ecm.Component<gz::sim::components::Inertial>(link);
                    if (inertialComp) {
                        total_mass += inertialComp->Data().MassMatrix().Mass();
                    }
                }

                if (total_mass > 0.0) {
                    double F_hover = total_mass * gravity_z;
                    this->hover_wrench_.mutable_force()->set_z(F_hover);
                    RCLCPP_INFO(this->rclcpp_node_->get_logger(),
                                "Fisica inizializzata: Massa=%.3f kg, Gravità=%.3f m/s^2, Forza Hovering=%.3f N",
                                total_mass, gravity_z, F_hover);
                } else {
                    this->hover_wrench_.mutable_force()->set_z(1.28 * 9.81); 
                    RCLCPP_WARN(this->rclcpp_node_->get_logger(), "Massa non trovata, uso fallback (1.28 kg).");
                }

                this->physics_initialized_ = true;
            }
        
            gz::msgs::Wrench wrench_to_use;
            {
                std::lock_guard<std::mutex> lock(this->wrench_commands_mutex_);
                
                // SE NESSUN COMANDO E' ANCORA ARRIVATO: RESTA A TERRA
                if (!this->first_wrench_received_) {
                    wrench_to_use.mutable_force()->set_x(0.0);
                    wrench_to_use.mutable_force()->set_y(0.0);
                    wrench_to_use.mutable_force()->set_z(0.0);
                    wrench_to_use.mutable_torque()->set_x(0.0);
                    wrench_to_use.mutable_torque()->set_y(0.0);
                    wrench_to_use.mutable_torque()->set_z(0.0);
                } 
                else 
                {
                    // IL CONTROLLORE HA INIZIATO A TRASMETTERE: VALUTA TIMEOUT
                    auto now_steady = std::chrono::steady_clock::now();
                    double dt_last_cmd = std::chrono::duration<double>(now_steady - this->last_cmd_time_).count();
                    
                    bool now_fallback = (dt_last_cmd > this->cmd_timeout_sec_);
                    
                    if (now_fallback)
                    {
                        wrench_to_use = this->hover_wrench_; // Applica Hovering
                        if (!this->in_fallback_)
                        {
                            this->fallback_enter_time_ = now_steady;
                            RCLCPP_WARN(this->rclcpp_node_->get_logger(),
                                        "Timeout comandi (%.2f s). ENTRA in FALLBACK (hover).",
                                        dt_last_cmd);
                        }
                    }
                    else
                    {
                        wrench_to_use = this->latest_wrench_command_; // Segui l'MPC
                        if (this->in_fallback_)
                        {
                            double fallback_dur = std::chrono::duration<double>(now_steady - this->fallback_enter_time_).count();
                            RCLCPP_INFO(this->rclcpp_node_->get_logger(),
                                        "Comandi wrench RIPRESI. Durata fallback: %.3f s.",
                                        fallback_dur);
                        }
                    }
                    this->in_fallback_ = now_fallback;
                }
            }
        
            Eigen::Vector4d desired_wrench_subset;
            desired_wrench_subset << wrench_to_use.force().z(),
                                     wrench_to_use.torque().x(),
                                     wrench_to_use.torque().y(),
                                     wrench_to_use.torque().z();
        
            Eigen::Vector4d rotor_speed_sq_commands = this->M_alloc_inv_ * desired_wrench_subset;
        
            gz::msgs::Actuators actuators_msg;
            for (size_t i = 0; i < this->rotors_.size(); ++i)
            {
                double rotor_speed_sq = rotor_speed_sq_commands(i);
                double target_angular_velocity = std::max(0.0, std::sqrt(rotor_speed_sq));
                actuators_msg.add_velocity(target_angular_velocity);
            }
        
            this->motor_speed_pub_.Publish(actuators_msg);
        }

    private: 
        void OnWrenchCommandsGz(const gz::msgs::Wrench &msg) 
        {
            std::lock_guard<std::mutex> lock(this->wrench_commands_mutex_);
            this->latest_wrench_command_ = msg;
            this->last_cmd_time_ = std::chrono::steady_clock::now();
            if (!this->first_wrench_received_) {
                this->first_wrench_received_ = true;
                RCLCPP_INFO(this->rclcpp_node_->get_logger(), "Ricevuto il PRIMO wrench esterno (Decollo autorizzato).");
            }
        }

        rclcpp::Node::SharedPtr rclcpp_node_;
        gz::sim::Model model_;
        std::vector<Rotor> rotors_;

        double cf_;
        double ct_;
        double L_;
        Eigen::Matrix4d M_alloc_inv_;

        gz::msgs::Wrench latest_wrench_command_;
        std::mutex wrench_commands_mutex_;

        gz::transport::Node gz_node_;
        gz::transport::Node::Publisher motor_speed_pub_;
        
        gz::msgs::Wrench hover_wrench_;
        std::chrono::steady_clock::time_point last_cmd_time_;
        double cmd_timeout_sec_ = 1;
        bool in_fallback_ = false; // Inizialmente falso perché siamo in stato IDLE, non FALLBACK
        bool first_wrench_received_ = false;
        std::chrono::steady_clock::time_point fallback_enter_time_;
        
        bool physics_initialized_;
    };
}

IGNITION_ADD_PLUGIN(mrsim_gazebo_sim::MrsimDroneIgnPlugin,
                    gz::sim::System,
                    gz::sim::ISystemConfigure, 
                    gz::sim::ISystemPreUpdate)