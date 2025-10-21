#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/components/Pose.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/transport/Node.hh>
#include <ignition/msgs/pose.pb.h>
#include <ignition/plugin/Register.hh>

namespace mrsim_gazebo_sim
{
  class SetPegPosePlugin : public ignition::gazebo::System,
                           public ignition::gazebo::ISystemConfigure,
                           public ignition::gazebo::ISystemPreUpdate
  {
    public: void Configure(const ignition::gazebo::Entity &_entity,
                           const std::shared_ptr<const sdf::Element> &,
                           ignition::gazebo::EntityComponentManager &_ecm,
                           ignition::gazebo::EventManager &) override
    {
      this->model = ignition::gazebo::Model(_entity);
      this->pose = ignition::math::Pose3d(2, 2, 0, 0, 0, 0);

      // Sottoscrivi al topic dove il bridge pubblica la pose
      this->ignNode.Subscribe("/model/my_peg/pose", &SetPegPosePlugin::OnPoseMsg, this);
    }

    public: void PreUpdate(const ignition::gazebo::UpdateInfo &_info,
                           ignition::gazebo::EntityComponentManager &_ecm) override
    {
      if (_info.paused)
        return;

      // Aggiorna la posa con l'ultima ricevuta dal subscriber
      // Qui viene mosso in gazebo l'oggetto
      this->model.SetWorldPoseCmd(_ecm, this->pose);
    }


  // Nel contesto della classe SetPegPosePlugin, ad esempio:

  void OnPoseMsg(const ignition::msgs::Pose &_msg)
  {
    // Estrai la posizione
    const auto &pos = _msg.position();
    ignition::math::Vector3d position(pos.x(), pos.y(), pos.z());
  
    // Estrai l'orientazione (quaternion)
    const auto &ori = _msg.orientation();
    ignition::math::Quaterniond rotation(ori.x(), ori.y(), ori.z(), ori.w());
  
    // Imposta la pose (position + rotation) sull'oggetto
    this->pose.Set(position, rotation);
  }

    private:
      ignition::gazebo::Model model{ignition::gazebo::kNullEntity};
      ignition::math::Pose3d pose;
      ignition::transport::Node ignNode {};
  };
}

IGNITION_ADD_PLUGIN(mrsim_gazebo_sim::SetPegPosePlugin,
                    ignition::gazebo::System,
                    ignition::gazebo::ISystemConfigure,
                    ignition::gazebo::ISystemPreUpdate)

IGNITION_ADD_PLUGIN_ALIAS(mrsim_gazebo_sim::SetPegPosePlugin,
                          "mrsim_gazebo_sim::SetPegPosePlugin")
