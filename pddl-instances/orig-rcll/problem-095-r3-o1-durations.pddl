(define (problem rcll-production-095-durative)
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
   (order-cap-color o1 CAP_BLACK)
   (order-gate o1 GATE-2)



   (= (path-length C-BS INPUT C-BS OUTPUT) 2.705076)
   (= (path-length C-BS INPUT C-CS1 INPUT) 9.639589)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 11.256088)
   (= (path-length C-BS INPUT C-CS2 INPUT) 6.239091)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 3.818097)
   (= (path-length C-BS INPUT C-DS INPUT) 6.911376)
   (= (path-length C-BS INPUT C-DS OUTPUT) 7.172992)
   (= (path-length C-BS OUTPUT C-BS INPUT) 2.705076)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 9.752541)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 11.369040)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 6.352043)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 3.931049)
   (= (path-length C-BS OUTPUT C-DS INPUT) 7.024328)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 7.285943)
   (= (path-length C-CS1 INPUT C-BS INPUT) 9.639590)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 9.752542)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 2.594709)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 5.580522)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 8.497086)
   (= (path-length C-CS1 INPUT C-DS INPUT) 8.847653)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 6.442599)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 11.256087)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 11.369040)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 2.594709)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 7.197020)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 10.113583)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 10.135415)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 8.059097)
   (= (path-length C-CS2 INPUT C-BS INPUT) 6.239091)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 6.352043)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 5.580521)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 7.197019)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 4.894356)
   (= (path-length C-CS2 INPUT C-DS INPUT) 5.244925)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 2.839869)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 3.818097)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 3.931049)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 8.497086)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 10.113585)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 4.894356)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 4.289989)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 4.551604)
   (= (path-length C-DS INPUT C-BS INPUT) 6.911376)
   (= (path-length C-DS INPUT C-BS OUTPUT) 7.024328)
   (= (path-length C-DS INPUT C-CS1 INPUT) 8.847655)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 10.135415)
   (= (path-length C-DS INPUT C-CS2 INPUT) 5.244926)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 4.289989)
   (= (path-length C-DS INPUT C-DS OUTPUT) 3.164009)
   (= (path-length C-DS OUTPUT C-BS INPUT) 7.172991)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 7.285943)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 6.442599)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 8.059096)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 2.839869)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 4.551604)
   (= (path-length C-DS OUTPUT C-DS INPUT) 3.164009)
   (= (path-length START INPUT C-BS INPUT) 2.845128)
   (= (path-length START INPUT C-BS OUTPUT) 1.955897)
   (= (path-length START INPUT C-CS1 INPUT) 8.704997)
   (= (path-length START INPUT C-CS1 OUTPUT) 10.321496)
   (= (path-length START INPUT C-CS2 INPUT) 5.304499)
   (= (path-length START INPUT C-CS2 OUTPUT) 2.883505)
   (= (path-length START INPUT C-DS INPUT) 5.976784)
   (= (path-length START INPUT C-DS OUTPUT) 6.238399))

  (:goal (order-fulfilled o1))
)
