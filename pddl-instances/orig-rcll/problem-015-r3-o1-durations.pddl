(define (problem rcll-production-015-durative)
	(:domain rcll-production-durative)
    
  (:objects
    R-1 - robot
    R-2 - robot
    R-3 - robot
    o1 - order
    wp1 - workpiece
    cg1 cg2 cg3 cb1 cb2 cb3 - cap-carrier
    C-BS C-CS1 C-CS2 C-DS - mps
    CYAN - team-color
  )
   
  (:init
   (mps-type C-BS BS)
   (mps-type C-CS1 CS)
   (mps-type C-CS2 CS) 
   (mps-type C-DS DS)
   (location-free START INPUT)
   (location-free C-BS INPUT)
   (location-free C-BS OUTPUT)
   (location-free C-CS1 INPUT)
   (location-free C-CS1 OUTPUT)
   (location-free C-CS2 INPUT)
   (location-free C-CS2 OUTPUT)
   (location-free C-DS INPUT)
   (location-free C-DS OUTPUT)
   (cs-can-perform C-CS1 CS_RETRIEVE)
   (cs-can-perform C-CS2 CS_RETRIEVE)
   (cs-free C-CS1)
   (cs-free C-CS2)

   (wp-base-color wp1 BASE_NONE)
   (wp-cap-color wp1 CAP_NONE)
   (wp-ring1-color wp1 RING_NONE)
   (wp-ring2-color wp1 RING_NONE)
   (wp-ring3-color wp1 RING_NONE)
   (wp-unused wp1)
   (robot-waiting R-1)
   (robot-waiting R-2)
   (robot-waiting R-3)

   (mps-state C-BS IDLE)
   (mps-state C-CS1 IDLE)
   (mps-state C-CS2 IDLE)
   (mps-state C-DS IDLE)

   (wp-cap-color cg1 CAP_GREY)
   (wp-cap-color cg2 CAP_GREY)
   (wp-cap-color cg3 CAP_GREY)
   (wp-on-shelf cg1 C-CS1 LEFT)
   (wp-on-shelf cg2 C-CS1 MIDDLE)
   (wp-on-shelf cg3 C-CS1 RIGHT)

   (wp-cap-color cb1 CAP_BLACK)
   (wp-cap-color cb2 CAP_BLACK)
   (wp-cap-color cb3 CAP_BLACK)
   (wp-on-shelf cb1 C-CS2 LEFT)
   (wp-on-shelf cb2 C-CS2 MIDDLE)
   (wp-on-shelf cb3 C-CS2 RIGHT)
   (order-complexity o1 c0)
   (order-base-color o1 BASE_RED)
   (order-cap-color o1 CAP_GREY)
   (order-gate o1 GATE-3)



   (= (path-length C-BS INPUT C-BS OUTPUT) 2.763203)
   (= (path-length C-BS INPUT C-CS1 INPUT) 11.017211)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 8.697878)
   (= (path-length C-BS INPUT C-CS2 INPUT) 5.412367)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 4.024849)
   (= (path-length C-BS INPUT C-DS INPUT) 7.386528)
   (= (path-length C-BS INPUT C-DS OUTPUT) 5.309017)
   (= (path-length C-BS OUTPUT C-BS INPUT) 2.763202)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 11.890197)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 9.491528)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 6.206017)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 4.897834)
   (= (path-length C-BS OUTPUT C-DS INPUT) 7.389635)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 6.102667)
   (= (path-length C-CS1 INPUT C-BS INPUT) 11.017212)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 11.890198)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.932367)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 6.990191)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 7.353044)
   (= (path-length C-CS1 INPUT C-DS INPUT) 6.108829)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 7.081307)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 8.697878)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 9.491528)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.932367)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 4.379828)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 5.903808)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 5.730980)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 4.470943)
   (= (path-length C-CS2 INPUT C-BS INPUT) 5.412367)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 6.206017)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 6.990191)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 4.379828)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 4.623616)
   (= (path-length C-CS2 INPUT C-DS INPUT) 4.248405)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 1.185433)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 4.024849)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 4.897834)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 7.353043)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 5.903808)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 4.623616)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 5.974768)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 4.714732)
   (= (path-length C-DS INPUT C-BS INPUT) 7.386528)
   (= (path-length C-DS INPUT C-BS OUTPUT) 7.389635)
   (= (path-length C-DS INPUT C-CS1 INPUT) 6.108829)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 5.730981)
   (= (path-length C-DS INPUT C-CS2 INPUT) 4.248405)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 5.974769)
   (= (path-length C-DS INPUT C-DS OUTPUT) 4.329829)
   (= (path-length C-DS OUTPUT C-BS INPUT) 5.309017)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 6.102667)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 7.081307)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 4.470943)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 1.185433)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 4.714732)
   (= (path-length C-DS OUTPUT C-DS INPUT) 4.329829)
   (= (path-length START INPUT C-BS INPUT) 1.577976)
   (= (path-length START INPUT C-BS OUTPUT) 3.939873)
   (= (path-length START INPUT C-CS1 INPUT) 10.200591)
   (= (path-length START INPUT C-CS1 OUTPUT) 7.881258)
   (= (path-length START INPUT C-CS2 INPUT) 4.595747)
   (= (path-length START INPUT C-CS2 OUTPUT) 3.208229)
   (= (path-length START INPUT C-DS INPUT) 6.569907)
   (= (path-length START INPUT C-DS OUTPUT) 4.492397))

  (:goal (order-fulfilled o1))
)
