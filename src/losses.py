# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    # TODO: 정답 클래스 확률의 log 값을 이용해 batch 평균 cross entropy를 계산하세요.
    # 힌트: np.clip으로 log(0)을 피하고, np.arange(batch_size)로 정답 위치를 고릅니다.
    # raise NotImplementedError("cross_entropy_loss를 구현하세요.")
    # 1. 배치 사이즈 구하고
    # 2. np.clip으로 1e-7 제일작은값, 1.0제일큰값 넣어서 log(0) 계속 무한히 작아지는 현상 막기 
    # 3. np.arange함수로 정답레이블의 손실함수 배열 구하기 
    # 4. 손실함수배열에 제일 손실많이 난거 로그를 씌운다. 
    batch_size = y_pred.shape[0]
    y_pred = np.clip(y_pred, 1e-7, 1.0)
    correct_probs = y_pred[np.arange(batch_size), y_true]
    loss = -np.mean(np.log(correct_probs))

    return loss 

    