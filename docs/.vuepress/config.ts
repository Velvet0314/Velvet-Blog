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
          avatar: 'https://image.velvet-notes.org/blog/avatar.png',
          url: 'https://github.com/Velvet0314'
        }
      ]
    }, 

    codeHighlighter: {
      themes: { light: 'solarized-light', dark: 'solarized-dark'} 
    },

    copyright: 'CC-BY-NC-SA-4.0',

    docsRepo: 'Velvet0314/Velvet-Blog',
    docsBranch: 'master',
    docsDir: 'docs',
    markdown: {
      annotation: true, 
      pdf: true,
    },
    plugins: {
        
      /**
       * markdown enhance
       * @see https://theme-plume.vuejs.press/config/plugins/markdown-enhance/
       */
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
      },
    },
  }),
})
