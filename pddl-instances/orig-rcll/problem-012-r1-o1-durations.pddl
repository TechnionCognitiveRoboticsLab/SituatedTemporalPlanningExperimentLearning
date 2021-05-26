(define (problem rcll-production-012-durative)
	(:domain rcll-production-durative)
    
  (:objects
    R-1 - robot
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
   (order-cap-color o1 CAP_BLACK)
   (order-gate o1 GATE-2)



   (= (path-length C-BS INPUT C-BS OUTPUT) 2.379678)
   (= (path-length C-BS INPUT C-CS1 INPUT) 8.045044)
   (= (path-length C-BS INPUT C-CS1 OUTPUT) 7.416770)
   (= (path-length C-BS INPUT C-CS2 INPUT) 5.163217)
   (= (path-length C-BS INPUT C-CS2 OUTPUT) 7.296554)
   (= (path-length C-BS INPUT C-DS INPUT) 7.838376)
   (= (path-length C-BS INPUT C-DS OUTPUT) 6.253968)
   (= (path-length C-BS OUTPUT C-BS INPUT) 2.379678)
   (= (path-length C-BS OUTPUT C-CS1 INPUT) 9.791262)
   (= (path-length C-BS OUTPUT C-CS1 OUTPUT) 9.162990)
   (= (path-length C-BS OUTPUT C-CS2 INPUT) 4.056141)
   (= (path-length C-BS OUTPUT C-CS2 OUTPUT) 6.189478)
   (= (path-length C-BS OUTPUT C-DS INPUT) 8.261720)
   (= (path-length C-BS OUTPUT C-DS OUTPUT) 6.201758)
   (= (path-length C-CS1 INPUT C-BS INPUT) 8.045045)
   (= (path-length C-CS1 INPUT C-BS OUTPUT) 9.791264)
   (= (path-length C-CS1 INPUT C-CS1 OUTPUT) 2.638202)
   (= (path-length C-CS1 INPUT C-CS2 INPUT) 7.863415)
   (= (path-length C-CS1 INPUT C-CS2 OUTPUT) 7.840217)
   (= (path-length C-CS1 INPUT C-DS INPUT) 6.738069)
   (= (path-length C-CS1 INPUT C-DS OUTPUT) 6.521048)
   (= (path-length C-CS1 OUTPUT C-BS INPUT) 7.416771)
   (= (path-length C-CS1 OUTPUT C-BS OUTPUT) 9.162991)
   (= (path-length C-CS1 OUTPUT C-CS1 INPUT) 2.638202)
   (= (path-length C-CS1 OUTPUT C-CS2 INPUT) 7.235141)
   (= (path-length C-CS1 OUTPUT C-CS2 OUTPUT) 7.211943)
   (= (path-length C-CS1 OUTPUT C-DS INPUT) 6.109796)
   (= (path-length C-CS1 OUTPUT C-DS OUTPUT) 5.892774)
   (= (path-length C-CS2 INPUT C-BS INPUT) 5.163217)
   (= (path-length C-CS2 INPUT C-BS OUTPUT) 4.056141)
   (= (path-length C-CS2 INPUT C-CS1 INPUT) 7.863415)
   (= (path-length C-CS2 INPUT C-CS1 OUTPUT) 7.235141)
   (= (path-length C-CS2 INPUT C-CS2 OUTPUT) 3.082638)
   (= (path-length C-CS2 INPUT C-DS INPUT) 4.220304)
   (= (path-length C-CS2 INPUT C-DS OUTPUT) 2.159628)
   (= (path-length C-CS2 OUTPUT C-BS INPUT) 7.296554)
   (= (path-length C-CS2 OUTPUT C-BS OUTPUT) 6.189478)
   (= (path-length C-CS2 OUTPUT C-CS1 INPUT) 7.840217)
   (= (path-length C-CS2 OUTPUT C-CS1 OUTPUT) 7.211943)
   (= (path-length C-CS2 OUTPUT C-CS2 INPUT) 3.082638)
   (= (path-length C-CS2 OUTPUT C-DS INPUT) 2.293123)
   (= (path-length C-CS2 OUTPUT C-DS OUTPUT) 2.136430)
   (= (path-length C-DS INPUT C-BS INPUT) 7.838376)
   (= (path-length C-DS INPUT C-BS OUTPUT) 8.261719)
   (= (path-length C-DS INPUT C-CS1 INPUT) 6.738070)
   (= (path-length C-DS INPUT C-CS1 OUTPUT) 6.109796)
   (= (path-length C-DS INPUT C-CS2 INPUT) 4.220304)
   (= (path-length C-DS INPUT C-CS2 OUTPUT) 2.293123)
   (= (path-length C-DS INPUT C-DS OUTPUT) 3.274097)
   (= (path-length C-DS OUTPUT C-BS INPUT) 6.253968)
   (= (path-length C-DS OUTPUT C-BS OUTPUT) 6.201757)
   (= (path-length C-DS OUTPUT C-CS1 INPUT) 6.521047)
   (= (path-length C-DS OUTPUT C-CS1 OUTPUT) 5.892774)
   (= (path-length C-DS OUTPUT C-CS2 INPUT) 2.159628)
   (= (path-length C-DS OUTPUT C-CS2 OUTPUT) 2.136431)
   (= (path-length C-DS OUTPUT C-DS INPUT) 3.274098)
   (= (path-length START INPUT C-BS INPUT) 0.890284)
   (= (path-length START INPUT C-BS OUTPUT) 3.133538)
   (= (path-length START INPUT C-CS1 INPUT) 7.670320)
   (= (path-length START INPUT C-CS1 OUTPUT) 7.042046)
   (= (path-length START INPUT C-CS2 INPUT) 5.101566)
   (= (path-length START INPUT C-CS2 OUTPUT) 7.234903)
   (= (path-length START INPUT C-DS INPUT) 7.463651)
   (= (path-length START INPUT C-DS OUTPUT) 6.192317))

  (:goal (order-fulfilled o1))
)
