---
title:  "[Optimization] Conjugate Gradients"
date:   2020-03-01 12:04:36 +0900
tags:
  - Math
---

이 포스팅에서는 $$Ax = b$$의 해를 찾아내는 과정을, 최적화 문제로 환원하여 해결하는 **conjugate gradients method**를 살펴봅니다. 
Justin Solomon 교수님의 [CS 205a: Mathematical Methods for Robotics, Vision, and Graphics][cs205a] 강의를 기본으로 하되 흐름에 맞게 내용들을 추가하고 재배치했음을 밝힙니다. 부족한점에 대한 지적과 질문은 자유롭게 댓글로 남겨주세요.

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

## Introduction

우리가 쉽게 마주하는 연립일차방정식은 중학교때 배우는 풀이법을 활용하여 해결할 수 있지만, 행렬을 활용하면 더욱 쉽게 답을 찾아낼 수 있습니다. <br>
연립일차방정식은 선형 방정식 $$Ax = b $$의 형태로 나타낼 수 있고, Gaussain elimination을 활용한다면 $$n$$개의 연립방정식에 대하여 언제나 $$O(n^3)$$의 시간안에 해를 구할 수 있습니다. 

하지만, 만약 저희가 풀고자 하는 문제의 $$n$$의 개수가 너무나 크다면 $$O(n^3)$$의 계산 시간을 기다리는 것은 사실상 불가능 할 것입니다.  
하지만, 공대생이라면 주어진 시간이 부족하다고 문제를 포기할 수는 없습니다. 정확한 해를 구하는 것이 불가능하다면, 주어진 시간동안 최대한 해에 가까운 값이라도 추정을 해내야 합니다.

Conjugate gradient method는 이러한 상황에서 적용하기 좋은 알고리즘입니다. 선형방정식 $$Ax=b$$의 해를 구하는 문제를 최적화 문제로 환원하여 해를 반복적으로 근사하되, $$O(n^3)$$시간 안에는 정확한 해를 찾을 수 있는 알고리즘입니다.
Conjugate gradient method가 정확히 무엇이고, 어떻게 구현할 수 있는지 천천히 알아보도록 하겠습니다.

## Gradient Descent

선형 방정식 $$Ax = b $$를 최적화 문제로 바꾸기에 앞서, conjugate gradient를 적용하기 위한 조건들에 대해 먼저 살펴보겠습니다.

- $$ A \in \mathbb{R}^{n \times n}$$ square matrix
- $$ A $$ is positive definite matrix

이런 조건들을 만족하는 matrix로는 identity matrix, invertible matrix 등이 있습니다. 
저는 강화학습 알고리즘중 하나인 Trust region policy optimization(TRPO)에서 효율적인 계산을 위해 conjugate gradient method를 공부하게 되었습니다.
주어진 positive definite matrix $$F$$에 대해서 $$F^{-1}b$$의 값을 구해야 하는데, $$F^{-1}$$을 구하는데 시간이 너무나 오래걸리는 것이 문제였습니다.
따라서, $$F^{-1}$$을 직접 구하는 것이 아니라, $$Fx = b$$의 해를 근사하는 방법으로 $$F^{-1}b$$의 값을 근사하는 것입니다.

### Linear equation to optimization

$$A\vec{x} = \vec{b}$$의 해를 조금만 자세히 살펴보면 아래의 방정식 $$f(\vec{x})$$의 최솟값의 해와 동일하다는 것을 쉽게 확인할 수 있습니다.

$$f(\vec{x}) = \cfrac{1}{2}\vec{x}^TA\vec{x} - \vec{b}^T\vec{x} + c$$

$$f(\vec{x})$$의 최솟값은 c값에 상관없이, 미분 값이 0인 지점이 되기 때문에

$$\nabla f(\vec{x}) = A\vec{x} - \vec{b},$$

에서 $$\nabla f(\vec{x}) = 0$$의 해는 $$A\vec{x} = \vec{b}$$의 해와 동일하게 됩니다.

이제 $$A\vec{x} = \vec{b}$$의 해를 구하는 문제가 $$f(\vec{x})$$의 최솟값을 구하는 문제로 바뀔 수 있는 것을 알았으니, 
$$\nabla f(\vec{x}) = 0$$을 직접적으로 찾는 것이 아니라, gradient descent 방법을 통해서 최적화를 진행할 수 있습니다.
Gradient descent 알고리즘을 간단하게 돌아보면 아래와 같습니다.

1. update 방향을 찾는다. $$\vec{d_k} = -\nabla f(\vec{x_{k-1}}) = \vec{b} - A\vec{x_{k-1}}$$
2. 추정값을 update한다. $$\vec{x_k} = \vec{x_{k-1}} + \alpha_k \vec{d_k} $$

### Determining step size

일반적으로 $$f$$를 $$\vec{d_k}$$방향에서 최소화 시킬 수 있는 이동의 크기 $$\alpha_k$$를 찾는 것은 쉬운일이 아니기 때문에, line search등의 방법을 통해 $$\alpha_k$$의 값을 찾곤 합니다.
하지만, 위와 같은 형태의 이차함수 $$f(\vec{x}) = \frac{1}{2}\vec{x}^TA\vec{x} - \vec{b}^T\vec{x} + c$$에선 $$\vec{d_k}$$방향으로 $$f$$를 최소화시키는 $$\alpha_k$$의 값을 직접 어렵지않게 계산할 수 있습니다.

Gradient descent가 이루어진 $$f$$의 값은 $$g(\alpha) = f(\vec{x} + \alpha \vec{d})$$이기 때문에 $$g(\alpha)$$를 최소화하는 $$\alpha$$의 값을 찾으면 됩니다.

<div>
$$
\begin{align*}
  g(\alpha) &= f(\vec{x} + \alpha \vec{d})\\[10pt]
 &= \frac{1}{2}(\vec{x} + \alpha \vec{d})^TA(\vec{x} + \alpha \vec{d}) - \vec{b}^T(\vec{x} + \alpha \vec{d}) + c\\[8pt]
 &= \frac{1}{2}(\vec{x}^TA\vec{x} + 2\alpha\vec{x}^TA\vec{d} + \alpha^{2}\vec{d}^TA\vec{d}) - \vec{b}^T\vec{x} + \vec{b}^T \alpha \vec{d} + c\\[8pt]
 &= \frac{1}{2}\alpha^{2}\vec{d}^TA\vec{d} + \alpha(\vec{x}^TA\vec{d} + \vec{b}^T \vec{d}) + const. \\[6pt]
\end{align*}
$$
<br></div>

이제 $$g(\alpha)$$를 $$\alpha$$에 대해서 최소화 하고 싶다면, $$\cfrac{dg}{d\alpha}(\alpha) = 0 $$이 되는 $$\alpha$$를 찾으면 됩니다.

$$ \frac{dg}{d\alpha}(\alpha) = \alpha\vec{d}^TA\vec{d} + \vec{d}^T(A\vec{x} - \vec{b}) = 0$$

$$ \alpha = \frac{\vec{d}^T(\vec{b} - A\vec{x})}{\vec{d}^TA\vec{d}} $$

그럼 이렇게 찾은 $$\alpha$$의 값을 저희의 gradient descent 알고리즘에 적용하면 아래와 같습니다.

1. update 방향을 찾는다. $$\vec{d_k} = \vec{b} - A\vec{x_{k-1}}$$
2. $$\vec{d_k}$$방향으로 $$f$$를 최소화 시키는 step size, $$ \alpha_k = \cfrac{\vec{d}^T_k (\vec{b} - A\vec{x_{k-1}})}{\vec{d}^T_k T A\vec{d_k}}$$를 구한다.
3. 추정 값 $$\vec{x_{k-1}}$$를 update한다. $$ \vec{x_k} = \vec{x_{k-1}} + \alpha_k \vec{d}_k $$

주어진 gradient descent 알고리즘의 수렴성과 수렴시간에 대한 자세한 내용은 cs 205a 강의의 note를 참조해주세요.

## Conjugate Gradient

비록, 위의 gradient descent는 local gradient $$\nabla f(\vec{x})$$의 방향으로 $$f$$를 최소화를 시켜주지만, 최솟값 $$\vec{x} = A^{-1}b$$으로 수렴하는데는 상당히 오랜 시간이 걸립니다.
$$n \times n$$ matrix $$A$$에 대하여 matrix-vector의 곱연산은 $$O(n^2)$$의 시간이 걸립니다. 
위의 gradient descent 알고리즘 역시 한번의 iteration마다 $$O(n^2)$$의 시간이 걸리게 됩니다. 따라서, iteration $$n$$번 이상 진행되어야 값이 수렴한다면 $$O(n^3)$$의 시간안에 정확한 해를 찾을 수 있는 Gaussian elimination을 적용하는 것이 오히려 나은 방법일 것입니다.
따라서, 저희가 원하는 알고리즘은 n번 안에 정확한 해를 찾을 수 있는 gradient descent 알고리즘, conjugate gradient descent 입니다.

### Quadratic optimization to Measuring distance

conjugate gradient 알고리즘은 저희의 이차함수 $$f(\vec{x})$$를 조금은 다른 시각에서 바라보며 발전되었습니다.
우선, 저희가 구하고자 하는 해 $$\vec{x}^*$$를 알고있다고 가정하겠습니다. 그렇다면, 자연스럽게 $$A\vec{x}^* = \vec{b}$$를 만족하게 됩니다.
이제 식을 $$\vec{x}^*$$를 이용하여 다시 전개해 보겠습니다.

<div>
$$
\begin{align*}
  f(\vec{x}) &= \cfrac{1}{2}\vec{x}^TA\vec{x} - \vec{b}^T\vec{x} + c \\[6pt]
 &= \cfrac{1}{2}(\vec{x}^TA\vec{x} - \vec{x}^T A \vec{x}^* - \vec{x}^{*T} A \vec{x} + \vec{x}^{*T}A\vec{x}^*) +  \cfrac{1}{2}(\vec{x}^T A \vec{x}^* + \vec{x}^{*T} A \vec{x} - \vec{x}^{*T}A\vec{x}^*) - \vec{b}^T\vec{x} + c \\[6pt]
 &= \cfrac{1}{2}(\vec{x} - \vec{x}^{*})^T A (\vec{x} - \vec{x}^*) + \cfrac{1}{2}(\vec{x}^T A \vec{x}^* + (A \vec{x}^{*})^T \vec{x} - \vec{x}^{*T}A\vec{x}^*) - \vec{b}^T\vec{x} + c\\[6pt]
 &= \cfrac{1}{2}(\vec{x} - \vec{x}^{*})^T A (\vec{x} - \vec{x}^*) + \vec{x}^T \vec{b} - \cfrac{1}{2}\vec{x}^{*T} \vec{b} - \vec{b}^T\vec{x} + c, \text{ since } A\vec{x}^* = \vec{b} \\[6pt]
 &= \cfrac{1}{2}(\vec{x} - \vec{x}^{*})^T A (\vec{x} - \vec{x}^*) - \cfrac{1}{2}\vec{x}^{*T} \vec{b} + c \\[6pt]
 &= \cfrac{1}{2}(\vec{x} - \vec{x}^{*})^T A (\vec{x} - \vec{x}^*) + const, \text{ since } \vec{x}^* \text{ is constant in terms of } \vec{x} \\[6pt]
\end{align*}
$$
<br></div>

저희는 앞에서 주어진 matrix $$A$$가 symmetric, postive definite matrix라고 가정한 바가 있습니다. 따라서, $$A = LL^T$$의 형태로 Cholesky분해를 할 수 있게 되고, 위의 수식을 아래의 형태로 전개할 수 있게 됩니다.

$$
\begin{align*}
  f(\vec{x}) &= \cfrac{1}{2}(\vec{x} - \vec{x}^{*})^T A (\vec{x} - \vec{x}^*) + const \\[6pt]
 &= \cfrac{1}{2}\vert \vert L^T(\vec{x} - \vec{x}^{*}) \vert \vert^2_2 + const \\[6pt]
\end{align*}
$$






















[cs205a]: https://graphics.stanford.edu/courses/cs205a-13-fall/index.html