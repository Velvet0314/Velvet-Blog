import { defineNoteConfig, defineNotesConfig } from 'vuepress-theme-plume'

const MachineLearning = defineNoteConfig({
  dir: 'ML',
  link: '/ML/',
  sidebar: ['', '第一章 机器学习入门','第二章 线性回归','第三章 二分类','第四章 广义线性模型','第五章 生成学习算法','第六章 支持向量机','第七章 学习理论','第八章 正则化与模型选择','第九章 EM算法','第十章 因子分析','第十一章 PCA 主成分分析','第十二章 ICA 独立成分分析','第十三章 强化学习'],
})

const DataStructure = defineNoteConfig({
  dir: 'DS',
  link: '/DS/',
  sidebar: ['','线性结构','树','图'],
})

export const notes = defineNotesConfig({
  dir: 'notes',
  link: '/',
  notes: [MachineLearning, DataStructure],
})