(define (problem rcll-production-093-durative)
	(:domain rcll-production-durative)
    
  (:objects
    R-1 - robot
    o1 - order
    wp1 - workpiece
    cg1 cg2 cg3 cb1 cb2 cb3 - cap-carrier
    C-BS C-CS1 C-CS2 C-DS - mps
    CYAN - team-color
  )
   
  (:init (order-delivery-window-open o1) (at 134.594 (not (order-delivery-window-open o1))) (can-commit-for-ontime-delivery o1) (at 6.30 (not (can-commit-for-ontime-delivery o1)))
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
   (order-gate o1 GATE-3)



   (= (path-length C-BS INPUT C-BS OUTPUT) 2.304431)
   (= (path-length C-BS INPUT C-CS1 INPUT) 8.487986)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 7.584125)
   (= (path-length C-BS INPUT C-CS2 INPUT) 1.939481)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 4.396661)
   (= (path-length C-BS INPUT C-DS INPUT) 6.702573)
   (= (path-length C-BS INPUT C-DS OUTPUT) 7.417700)
   (= (path-length C-BS OUTPUT C-BS INPUT) 2.304431)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 7.072862)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 6.169002)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 4.116001)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 5.122940)
   (= (path-length C-BS OUTPUT C-DS INPUT) 5.287449)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 7.340329)
   (= (path-length C-CS1 INPUT C-BS INPUT) 8.487987)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 7.072863)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 2.744020)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 8.277340)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 8.811565)
   (= (path-length C-CS1 INPUT C-DS INPUT) 6.708544)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 7.301968)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 7.584125)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 6.169001)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 2.744020)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 7.373478)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 7.748199)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 4.723475)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 5.316900)
   (= (path-length C-CS2 INPUT C-BS INPUT) 1.939481)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 4.116000)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 8.277339)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 7.373478)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 3.097141)
   (= (path-length C-CS2 INPUT C-DS INPUT) 6.491927)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 6.756919)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 4.396661)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 5.122940)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 8.811566)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 7.748199)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 3.097141)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 4.767296)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 3.890300)
   (= (path-length C-DS INPUT C-BS INPUT) 6.702573)
   (= (path-length C-DS INPUT C-BS OUTPUT) 5.287449)
   (= (path-length C-DS INPUT C-CS1 INPUT) 6.708544)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 4.723475)
   (= (path-length C-DS INPUT C-CS2 INPUT) 6.491927)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 4.767297)
   (= (path-length C-DS INPUT C-DS OUTPUT) 3.795453)
   (= (path-length C-DS OUTPUT C-BS INPUT) 7.417701)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 7.340329)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 7.301968)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 5.316899)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 6.756919)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 3.890300)
   (= (path-length C-DS OUTPUT C-DS INPUT) 3.795453)
   (= (path-length START INPUT C-BS INPUT) 3.483779)
   (= (path-length START INPUT C-BS OUTPUT) 1.325967)
   (= (path-length START INPUT C-CS1 INPUT) 6.668782)
   (= (path-length START INPUT C-CS1 OUTPUT) 5.764921)
   (= (path-length START INPUT C-CS2 INPUT) 4.184632)
   (= (path-length START INPUT C-CS2 OUTPUT) 4.718859)
   (= (path-length START INPUT C-DS INPUT) 4.883368)
   (= (path-length START INPUT C-DS OUTPUT) 6.936248))

  (:goal (order-fulfilled o1))
)
