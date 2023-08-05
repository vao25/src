# src
Ondřej Valtr; Bakalářská práce 2022/23 - SLAM; repozitář pro adresář src v ROS2 workspace

## Instalace
> mkdir ~/ros2_ws

> cd ~/ros2_ws

> git clone https://github.com/vao25/src.git

## Spuštění v ros2:

### EKF-SLAM
**ve workspace ros2_ws:**
```
colcon build --packages-select ekfSLAM
source install/setup.bash
ros2 run ekfSLAM ekf_sim
```

### FASTSLAM
**ve workspace ros2_ws:**
```
colcon build --packages-select fastslam
source install/setup.bash
ros2 run fastslam fast_sim
```

## Použité *"topics"*

*Pozn.: Ve finální verzi tyto topics nakonec použité nejsou. Uvedeno i pro jaké vytvořené zprávy jsou určeny, které jsou do nich publikovány (tyto zprávy nejsou určeny pro Rviz 2). Pro případ, kdyby např. bylo potřeba tyto data odebírat jinými uzly (node) aplikace (např. pomocí subscriberů). Jsou připraveny k použití odkomentováním části pro vytvoření jejich publisherů v konstruktoru třídy (EKF nebo FAST) a příkazů pro zahájení publikování v metodě main() v src/ekfSLAM/ekfSLAM/node.py (src/fastslam/fastlam/node.py) .*

*true* pro msgTrue

*path* pro msgPath

*stateX* pro msgStateX

*stateP* pro msgStateP (uděláno pouze pro EKF-SLAM)

*stateLen* pro msgStateLen

## Nastínění rekonstrukce dat z přijmutých zpráv:

```python
robotTrue = []
robotPath = []
for i in range((len(msgTrue)/3)):
    j = 3*i
    robotTrue.append([msgTrue[j], msgTrue[j+1], msgTrue[j+2]])
    robotPath.append([msgPath[j], msgPath[j+1], msgPath[j+2]])
    
X = []
P = [] # diag, done only for EKF-SLAM
j = 0
for i in range(len(msgStateLen)):
    k = int(msgStateLen[i])
    X.append([msgStateX[j:j+k]])
    P.append([msgStateP[j:j+k]]) # done only for EKF-SLAM
    j = j + k
```   
