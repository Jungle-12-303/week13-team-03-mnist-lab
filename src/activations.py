# -*- coding: utf-8 -*-
"""
활성화 함수 모음.

학생 구현 대상:
- ReLU.forward, ReLU.backward
- Softmax.forward, Softmax.backward
"""

import numpy as np

# 활성화 함수 사용하는 이유?
# 신경망에 비선형성을 넣어 주기 위해 사용한다
# 활성화 함수가 없으면 여러 Affine 계층을 쌓아도 하나의 선형 변환으로 합쳐지기 때문에 깊은 신경망이 복잡한 패턴을 표현할수 X
class ReLU:
    """
    ReLU(Rectified Linear Unit) 활성화 함수.

    은닉층에서 음수 값은 0으로 막고, 양수 값은 그대로 통과시킵니다.
    forward에서 만든 mask는 backward 때 "어느 위치로 gradient를 흘릴지" 결정하는 데 사용됩니다.
    """

    # -1 5
    # -3 2
    def forward(self, x):
        # self.mask
        # T F
        # T F
        self.mask = (x <= 0)

        #-1 5
        #-3 2
        result = x.copy()

        # result에서 mask가 true인 곳만 0으로 바꿔라
        # 0 5
        # 0 2
        
        result[self.mask] = 0

        # 0 5
        # 0 2
        return result
        """
        Args:
            x: 임의 shape의 입력 배열

        Returns:
            x와 같은 shape. x > 0인 위치만 원래 값을 유지합니다.
        """
        # TODO: x > 0 위치를 self.mask에 저장하고, 음수/0 위치는 0으로 바꾸세요.
        
        # raise NotImplementedError("ReLU.forward를 구현하세요.")

    def backward(self, dout):
        """
        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            ReLU 입력 x에 대한 gradient. forward 때 x <= 0이었던 위치는 0입니다.
        """
        # TODO: forward에서 저장한 self.mask를 이용해 gradient가 흐를 위치만 남기세요.
        # -1 5
        # -3 2

        # T F
        # T F
        dout[self.mask] = 0

        result = dout

        return result
        

        raise NotImplementedError("ReLU.backward를 구현하세요.")


class Softmax:
    """
    Softmax 출력층.

    각 샘플의 로짓(logit)을 클래스별 확률로 바꿉니다.
    exp 계산 전에 행별 최댓값을 빼면 큰 숫자에서 overflow가 나는 것을 줄일 수 있습니다.
    """

    #1, 3, 2
    #5, 4, 6
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_classes) 로짓

        Returns:
            (batch_size, num_classes) 확률. 각 행의 합은 1입니다.
        """
        # TODO: 수치 안정성을 위해 row별 max를 뺀 뒤 softmax 확률을 계산하세요.
        # 힌트: np.max(..., axis=1, keepdims=True), np.exp, np.sum을 사용합니다.
        # 3
        # 6
        c = np.max(x, axis = 1, keepdims = True)
        
        # -2 0 -1
        # -1 -2 0
        
        #[0.1353, 1.0000, 0.3679],
        #[0.3679, 0.1353, 1.0000]
        exp_x = np.exp(x - c)

        #  [1.5032],
        #  [1.5032]
        sum_exp_x = np.sum(exp_x, axis = 1, keepdims = True)

        #0.0900, 0.6652, 0.2447
        #0.2447, 0.0900, 0.6652
        y = exp_x / sum_exp_x

        return y


        raise NotImplementedError("Softmax.forward를 구현하세요.")

    def backward(self, dout):
        """
        Softmax와 Cross Entropy를 함께 미분한 gradient를 train()에서 직접 만들기 때문에
        여기서는 받은 gradient를 그대로 통과시킵니다.
        """
        # TODO: train()에서 만든 gradient를 그대로 반환하세요.
        return dout
        raise NotImplementedError("Softmax.backward를 구현하세요.")
