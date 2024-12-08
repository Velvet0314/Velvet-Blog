import { viteBundler } from '@vuepress/bundler-vite'
import { defineUserConfig } from 'vuepress'
import { plumeTheme } from 'vuepress-theme-plume'
import { path } from 'vuepress/utils'
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineUserConfig({
  base: '/',
  lang: 'zh-CN',
  title: 'Velvet-Notes',
  description: 'Test',
  bundler: viteBundler(),
  head: [
    ['link', { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/favicon-32x32.png' }],
  ],

  alias: {
    '@theme/VPBackToTop.vue': path.resolve(
      __dirname,
      './components/MyBackToTop.vue',
    ),},
  theme: plumeTheme({
    // 添加您的部署域名
    hostname: 'https://velvet-notes.org/',

    contributors: {
      mode: 'block',
      info: [
        {
          username: 'Velvet0314',
          name: 'Velvet',
          alias: ['李青阳'],
          avatar: 'https://s21.ax1x.com/2024/11/10/pA629aj.jpg',
          url: 'https://github.com/Velvet0314'
        }
      ]
    }, 

    plugins: {
      /**
       * Shiki 代码高亮
       * @see https://theme-plume.vuejs.press/config/plugins/code-highlight/
       */
      shiki: {
        languages: ['c','c++'],
        theme: { light: 'solarized-light', dark: 'solarized-dark'}
      },

      /**
       * markdown enhance
       * @see https://theme-plume.vuejs.press/config/plugins/markdown-enhance/
       */
      markdownEnhance: {
        demo: true,
      //   include: true,
      //   chart: true,
      //   echarts: true,
      //   mermaid: true,
      //   flowchart: true,
      },

      /**
       *  markdown power
       * @see https://theme-plume.vuejs.press/config/plugin/markdown-power/
       */
      // markdownPower: {
      //   pdf: true,
      //   caniuse: true,
      //   plot: true,
      //   bilibili: true,
      //   youtube: true,
      //   icons: true,
      //   codepen: true,
      //   replit: true,
      //   codeSandbox: true,
      //   jsfiddle: true,
      //   repl: {
      //     go: true,
      //     rust: true,
      //     kotlin: true,
      //   },
      // },

      /**
       * 评论 comments
       * @see https://theme-plume.vuejs.press/guide/features/comments/
       */
      comment: {
        provider: 'Giscus', // "Artalk" | "Twikoo" | "Waline"
        comment: true,
        repo: 'Velvet0314/Velvet-Blog',
        repoId: 'R_kgDONNA0PQ',
        category: 'Q&A',
        categoryId: 'DIC_kwDONNA0Pc4CkI6C',
        mapping: 'pathname',
        reactionsEnabled: true,
        inputPosition: 'top',
        lazyLoading: true,
        lightTheme: 'light',
        darkTheme: 'dark'
      },
    },
  }),
})
