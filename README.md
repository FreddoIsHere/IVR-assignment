# IVR-assignment
Prints target estimation and true target porsition to terminal. Target is apparently not present at the start of simulator. That is why it is printed in iteration 100. At iteration 400 joint angles are printed. Joint angles can be a little inaccurate because it still wiggles at iteration 400, improvements are needed: end effector/4th angle is not the right one in some cases.

## Joint angle estimation
(Note that I wrote "joint velocity estimatation" in the commit message, but I meant angle)
It seems unavoidable that at some points will be entirely or partially obscured. To counteract this I have made it so joint angles are estimated in these cases using the previous position and joint velocities. This occurs when a joint is obscured that we need to calculate the joint angle, and when there is a large difference in the joint angle and its estimated joint angles using the previous position and velocity. 
