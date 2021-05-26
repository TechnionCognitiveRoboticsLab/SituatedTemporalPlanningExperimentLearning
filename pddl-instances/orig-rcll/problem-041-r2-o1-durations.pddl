(define (problem rcll-production-041-durative)
	(:domain rcll-production-durative)
    
  (:objects
    R-1 - robot
    R-2 - robot
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
   (order-base-color o1 BASE_BLACK)
   (order-cap-color o1 CAP_BLACK)
   (order-gate o1 GATE-3)



   (= (path-length C-BS INPUT C-BS OUTPUT) 3.704706)
   (= (path-length C-BS INPUT C-CS1 INPUT) 7.191370)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 4.349762)
   (= (path-length C-BS INPUT C-CS2 INPUT) 11.249866)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 12.302854)
   (= (path-length C-BS INPUT C-DS INPUT) 7.638642)
   (= (path-length C-BS INPUT C-DS OUTPUT) 6.193671)
   (= (path-length C-BS OUTPUT C-BS INPUT) 3.704706)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 6.750029)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 4.138580)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 9.957742)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 12.091672)
   (= (path-length C-BS OUTPUT C-DS INPUT) 6.346517)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 4.901547)
   (= (path-length C-CS1 INPUT C-BS INPUT) 7.191370)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 6.750029)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 4.099987)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 5.714939)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 7.905962)
   (= (path-length C-CS1 INPUT C-DS INPUT) 4.016147)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 2.943033)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 4.349762)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 4.138580)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 4.099987)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 8.323610)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 9.211471)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 6.689431)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 5.244461)
   (= (path-length C-CS2 INPUT C-BS INPUT) 11.249865)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 9.957741)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 5.714938)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 8.323611)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 2.994935)
   (= (path-length C-CS2 INPUT C-DS INPUT) 6.859188)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 6.150746)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 12.302853)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 12.091671)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 7.905962)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 9.211471)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 2.994935)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 8.536113)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 8.341770)
   (= (path-length C-DS INPUT C-BS INPUT) 7.638642)
   (= (path-length C-DS INPUT C-BS OUTPUT) 6.346518)
   (= (path-length C-DS INPUT C-CS1 INPUT) 4.016147)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 6.689431)
   (= (path-length C-DS INPUT C-CS2 INPUT) 6.859188)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 8.536114)
   (= (path-length C-DS INPUT C-DS OUTPUT) 4.331247)
   (= (path-length C-DS OUTPUT C-BS INPUT) 6.193671)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 4.901547)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 2.943033)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 5.244460)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 6.150746)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 8.341770)
   (= (path-length C-DS OUTPUT C-DS INPUT) 4.331247)
   (= (path-length START INPUT C-BS INPUT) 2.150619)
   (= (path-length START INPUT C-BS OUTPUT) 3.737060)
   (= (path-length START INPUT C-CS1 INPUT) 5.840639)
   (= (path-length START INPUT C-CS1 OUTPUT) 2.999031)
   (= (path-length START INPUT C-CS2 INPUT) 9.899136)
   (= (path-length START INPUT C-CS2 OUTPUT) 10.952122)
   (= (path-length START INPUT C-DS INPUT) 6.287910)
   (= (path-length START INPUT C-DS OUTPUT) 4.842940))

  (:goal (order-fulfilled o1))
)
