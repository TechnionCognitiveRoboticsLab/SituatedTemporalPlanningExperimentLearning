(define (problem rcll-production-084-durative)
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
   (order-base-color o1 BASE_SILVER)
   (order-cap-color o1 CAP_GREY)
   (order-gate o1 GATE-1)



   (= (path-length C-BS INPUT C-BS OUTPUT) 2.178015)
   (= (path-length C-BS INPUT C-CS1 INPUT) 7.957573)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 5.649588)
   (= (path-length C-BS INPUT C-CS2 INPUT) 10.544927)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 11.880274)
   (= (path-length C-BS INPUT C-DS INPUT) 7.243809)
   (= (path-length C-BS INPUT C-DS OUTPUT) 6.130573)
   (= (path-length C-BS OUTPUT C-BS INPUT) 2.178015)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 5.820045)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 6.350342)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 9.700910)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 9.742746)
   (= (path-length C-BS OUTPUT C-DS INPUT) 7.944562)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 6.831326)
   (= (path-length C-CS1 INPUT C-BS INPUT) 7.957572)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 5.820045)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 3.096646)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 4.422513)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 6.188888)
   (= (path-length C-CS1 INPUT C-DS INPUT) 3.036451)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 2.399089)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 5.649588)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 6.350341)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 3.096646)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 5.491111)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 7.257485)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 2.783469)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 1.076756)
   (= (path-length C-CS2 INPUT C-BS INPUT) 10.544927)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 9.700909)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 4.422513)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 5.491111)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 2.853647)
   (= (path-length C-CS2 INPUT C-DS INPUT) 3.715380)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 4.793554)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 11.880274)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 9.742746)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 6.188888)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 7.257485)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 2.853647)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 6.351542)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 6.559929)
   (= (path-length C-DS INPUT C-BS INPUT) 7.243808)
   (= (path-length C-DS INPUT C-BS OUTPUT) 7.944562)
   (= (path-length C-DS INPUT C-CS1 INPUT) 3.036450)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 2.783468)
   (= (path-length C-DS INPUT C-CS2 INPUT) 3.715380)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 6.351542)
   (= (path-length C-DS INPUT C-DS OUTPUT) 3.218890)
   (= (path-length C-DS OUTPUT C-BS INPUT) 6.130572)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 6.831326)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 2.399089)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 1.076756)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 4.793553)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 6.559928)
   (= (path-length C-DS OUTPUT C-DS INPUT) 3.218890)
   (= (path-length START INPUT C-BS INPUT) 3.133430)
   (= (path-length START INPUT C-BS OUTPUT) 0.995903)
   (= (path-length START INPUT C-CS1 INPUT) 5.550431)
   (= (path-length START INPUT C-CS1 OUTPUT) 6.080727)
   (= (path-length START INPUT C-CS2 INPUT) 9.431295)
   (= (path-length START INPUT C-CS2 OUTPUT) 9.473132)
   (= (path-length START INPUT C-DS INPUT) 7.674948)
   (= (path-length START INPUT C-DS OUTPUT) 6.561712))

  (:goal (order-fulfilled o1))
)
