# CWNU-RDA
CWNU-RDA
1. Introduction to the laboratory

     Internet of things perception and big data analysis key Laboratory of Nanchong city was established in 2014 and designated by Nanchong as the city's key laboratory in 2019. The working area of the laboratory is more than 300 square meters. Experimental Equipment 61 sets, worth more than 410 million RMB. There are 39 scientific researchers, including 30 senior professional and technical personnel, 29 doctors and 3 national and provincial experts. Its research areas include: optimization theory and methods, distributed machine learning, Internet of things awareness based on RFID, animal recognition based on computer vision, intelligent information processing. It has presided over 53 national and provincial projects, won a natural science award from the Ministry of Education, and published more than 200 papers on SCI, EI and other topics, there are 20 papers about SCI and EI retrieval. The director of the laboratory is Professor Chen Yihong and the director of the Academic Committee is Professor Feng Quanyuan.

2. Data collection environment

Devices：

                Reader：Impinj R420

                Antenna：Larid S9028PCR

                Tag：Impinj M4E

Ground：

                3.5m (length) × 3.2m (width) × 4.5m (height)

Volunteers:

     This research team recruited fifteen volunteers, and the volunteers varied in sex ratio, height, and age. The ratio of male to female is 8:6, the height ranges from 1.55m to 1.85m, and the age ranges from 18 to 30 years. Each volunteer repeated each human activity 40 times, and each completed human activity is an activity sample. The time to complete an activity sample is also known as the human activity duration, and the duration was reasonably set to 5 seconds. There are a total of 1.26×104 activity samples when fifteen volunteers completed twenty-one activities.

3. An example of data

     Each tag response record in the dataset includes the EPC, timestamp, Dopler Frequency, RSSI, Phase, and the activity number (label) used to identify the activity.

Example:
                0010 1667632738934513 78.0625 -64.5 5.884350302329319 0
                0009 1667633437037975 -15.375 -77.5 4.424000592262189 1
                0008 1667634721907184 25.125 -74.5 1.9819031779482483 2
                0004 1667635276410115 -0.75 -74.5 0.19634954084936207 3
                0015 1667635970702292 -65.0625 -64.0 3.06182565261974 4
4. Types of activities

label	activity names	activity descriptions	label	activity names	activity descriptions
0	standing	Stand up straight	11	mark time with Swing arms	Swing your arms and march in place
1	Standing to crouching	Stand up straight at the beginning of the collection, squat at 3s, and crouch continuously thereafter	12	mark time without Swing arms	Walk in place without swinging arms
2	crouching	Crouching motionless	13	sit down	Stand up straight at the beginning of the collection, sit down at the third second, and then sit still
3	Crouching to standing	Start the collection by crouching, stand up for 3s, and then stand still	14	sitting	Sit still
4	Standing to stooping	Start the collection standing, bend over for the third second, then stay stooping	15	sitting to stooping	Sit up straight at the beginning of the collection, stoop down at the third second, and then remain seated and stooping
5	Standing and stooping	Stand and stoop still	16	sitting and stooping	Remain seated and stooping
6	stooping to standing	Stooping at the beginning of the collection, stand up straight for 3s, then stay upright	17	stooping to sitting	Sit and bend at the beginning of collection, sit upright at the third second, and then sit still
7	stride with swinging arms	Swing your arms in big strides	18	sitting to standing	Sit at the beginning of the acquisition, stand up at the third second, and then stay upright
8	steps with swing arms	Take small steps and swing your arms	19	lying	Lying motionless
9	stride without swinging arms	Take strides without swinging your arms	20	lying to sitting	Lying at the beginning of the collection, sit up at the third second, and then sit up straight
10	steps without swing arms	Take small steps without waving your arms	-	-	-
