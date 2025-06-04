import casadi as ca
import pinocchio as pin
import pinocchio.casadi_model as pin_cm

# 1️⃣ Build normal model
model = pin.buildModelFromUrdf("go2_original.urdf")

# 2️⃣ Build CasADi model
model_cd = pin_cm.buildModel(model)

# 3️⃣ Use model_cd with SX
q = ca.SX.sym("q", model_cd.nq)
data = model_cd.createData()

pin.forwardKinematics(model_cd, data, q)
pin.updateFramePlacements(model_cd, data)

frame_id = model_cd.getFrameId("FL_foot")
foot_pos = data.oMf[frame_id].translation
print("Foot position SX:", foot_pos)
